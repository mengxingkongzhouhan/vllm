# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from enum import Enum

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    LENGTH_AWARE = "length_aware"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    def begin_step(self, token_budget: int) -> None:
        """Hook called once by the scheduler at the start of each scheduling
        step, before the waiting queue is drained.

        Stateful policies (e.g. length-aware token-bucket aging) use it to
        refresh per-step state. No-op by default so FCFS/priority queues are
        unaffected.
        """
        return None


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def _default_length_fn(request: Request) -> int:
    """Fallback compute-length: prompt/decode length minus already-computed
    tokens. Used when no prefix-cache-aware ``length_fn`` is supplied."""
    return max(request.num_tokens - request.num_computed_tokens, 0)


class LengthAwareRequestQueue(RequestQueue):
    """Length-aware prefill scheduling queue.

    Inspired by vllm-ascend PR #10285 (LAPS), but with a key structural
    difference: **requests are NOT classified into short/long lanes at enqueue
    time.** Prompt-length is a poor proxy for cost once caching is involved, and
    -- more importantly -- how many tokens actually need to be computed depends
    on the *live* cache state (local prefix cache + external KV-connector
    cache), which changes over time and is only cheaply/safely knowable at
    scheduling time.

    Instead, classification happens lazily at **selection time**: whenever a
    request is about to be admitted to the running set, the queue scans all
    waiting requests, computes each one's real compute length via
    ``length_fn(request)`` (prompt minus local *and* remote cache hits), and
    partitions them into short (``compute_len <= threshold``) and long
    (``> threshold``). Dispatch priority is::

        immediate > aged-long > short > long

    - **immediate**: recovery-style requests (preempted / partially computed),
      dispatched first. Populated via ``prepend_request`` or the optional
      ``immediate_predicate``. These are held in a small side lane and are not
      re-scanned for length.
    - **short**: the oldest waiting request whose (live) compute length is
      ``<= threshold``.
    - **long**: everything else, dispatched only when no short is available or
      when the oldest long has aged past ``long_max_wait_ms``.

    Anti-starvation uses a token bucket: it refills by
    ``long_token_reservation * per-step token budget`` every ``begin_step`` and,
    once the oldest long has waited ``>= long_max_wait_ms``, that long may be
    promoted ahead of waiting shorts only while the bucket has credit. Admitting
    an aged long ahead of shorts spends its full remaining compute length from
    the bucket, so the aged-long admission rate stays bounded by the reservation
    fraction and shorts always retain ``1 - reservation`` of throughput.

    Requests (other than immediate) are held in a single FCFS structure so that
    "oldest" is well defined; this policy assumes a single total arrival order
    (like FCFS).

    Cost note: selection is O(N) in the number of waiting requests (it evaluates
    ``length_fn`` for each), so a scan is done per admitted request. The result
    of a ``peek_request`` is cached and reused by the immediately following
    ``pop_request`` (no queue mutation happens in between), and the cache is
    invalidated on any mutation and at ``begin_step``.

    NOTE: the bucket is charged inside ``pop_request``. The scheduler also pops
    from this queue on rare "skip" paths (e.g. a request that momentarily
    exceeds ``max_loras``); such a pop may count as an aged-long promotion. This
    is a deliberate, benign approximation -- an occasional extra charge only
    slows aged-long admission, which errs toward protecting short requests.
    """

    def __init__(
        self,
        length_fn: Callable[[Request], int] | None = None,
        threshold: int = 256,
        long_max_wait_ms: float = 0.0,
        long_token_reservation: float = 0.0,
        long_burst_steps: int = 4,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.length_fn = length_fn if length_fn is not None else _default_length_fn
        self.threshold = threshold
        self.long_max_wait_ms = long_max_wait_ms
        self.long_token_reservation = long_token_reservation
        self.long_burst_steps = long_burst_steps
        self.immediate_predicate = immediate_predicate

        # Recovery/preempted requests, dispatched first (not length-scanned).
        self._immediate = FCFSRequestQueue()
        # Everyone else, in arrival order. NO short/long split here -- that is
        # decided at selection time from the live compute length.
        self._waiting = FCFSRequestQueue()

        # req_id -> the lane currently holding the request.
        self._queue_index: dict[str, RequestQueue] = {}
        # req_id -> monotonic timestamp the request entered the waiting lane
        # (used for aging).
        self._enqueue_at: dict[str, float] = {}
        # req_ids explicitly forced into the immediate lane (recovery flows).
        self._force_immediate_ids: set[str] = set()

        # Token bucket throttling aged-long admissions (see begin_step).
        self._long_bucket = 0.0
        self._long_bucket_capacity = 0.0
        self._long_refill_per_step = 0.0

        # Cached selection: (request, charge_tokens). charge_tokens > 0 only for
        # an aged-long promotion that jumps waiting shorts. Invalidated on any
        # mutation and at begin_step.
        self._selection: tuple[Request, int] | None = None

    # ------------------------------------------------------------------ #
    # Per-step hook
    # ------------------------------------------------------------------ #
    def begin_step(self, token_budget: int) -> None:
        """Refill the aged-long admission bucket for this scheduling step."""
        self._long_refill_per_step = self.long_token_reservation * token_budget
        self._long_bucket_capacity = self._long_refill_per_step * self.long_burst_steps
        self._long_bucket = min(
            self._long_bucket_capacity,
            self._long_bucket + self._long_refill_per_step,
        )
        # Live cache state may have changed since last step, so any cached
        # selection (which depends on compute lengths) is now stale.
        self._selection = None

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _lane_for_enqueue(
        self, request: Request, force_immediate: bool
    ) -> RequestQueue:
        rid = request.request_id
        if force_immediate or rid in self._force_immediate_ids:
            return self._immediate
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate
        return self._waiting

    def _track(self, request: Request, lane: RequestQueue) -> None:
        rid = request.request_id
        self._queue_index[rid] = lane
        self._enqueue_at.setdefault(rid, time.monotonic())

    def _forget(self, request: Request, lane: RequestQueue) -> None:
        rid = request.request_id
        self._queue_index.pop(rid, None)
        self._enqueue_at.pop(rid, None)
        self._force_immediate_ids.discard(rid)

    # ------------------------------------------------------------------ #
    # Selection (scans all waiting requests each time)
    # ------------------------------------------------------------------ #
    def _compute_selection(self) -> tuple[Request, int] | None:
        # Immediate (recovery) requests always win and are not length-scanned.
        if self._immediate:
            return (self._immediate.peek_request(), 0)

        now = time.monotonic()
        oldest_short: Request | None = None
        oldest_long: Request | None = None
        oldest_long_len = 0
        has_short = False

        # Single pass in arrival order: the first short/long encountered is the
        # oldest short/long. compute length is evaluated live for every request.
        for request in self._waiting:
            compute_len = max(self.length_fn(request), 0)
            if compute_len <= self.threshold:
                has_short = True
                if oldest_short is None:
                    oldest_short = request
            elif oldest_long is None:
                oldest_long = request
                oldest_long_len = compute_len

        # Anti-starvation: the oldest long, once aged, may be promoted ahead of
        # waiting shorts, gated by the token-reservation bucket.
        if oldest_long is not None and self.long_max_wait_ms > 0:
            enqueued_at = self._enqueue_at.get(oldest_long.request_id, now)
            waited_ms = (now - enqueued_at) * 1000.0
            if waited_ms >= self.long_max_wait_ms:
                if not has_short:
                    # Stall avoidance: no short waiting, run the long (no charge).
                    return (oldest_long, 0)
                if self._long_bucket > 0.0:
                    # Reservation credit available: promote and charge its full
                    # remaining compute length.
                    return (oldest_long, oldest_long_len)
                # Bucket exhausted with shorts waiting: prefer shorts so the
                # aged-long admission rate stays bounded by the reservation.

        if oldest_short is not None:
            return (oldest_short, 0)
        if oldest_long is not None:
            return (oldest_long, 0)
        return None

    def _ensure_selection(self) -> tuple[Request, int] | None:
        if self._selection is None:
            self._selection = self._compute_selection()
        return self._selection

    # ------------------------------------------------------------------ #
    # RequestQueue interface
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request) -> None:
        lane = self._lane_for_enqueue(request, force_immediate=False)
        lane.add_request(request)
        self._track(request, lane)
        self._selection = None

    def pop_request(self) -> Request:
        selection = self._ensure_selection()
        if selection is None:
            raise IndexError("pop from an empty LengthAwareRequestQueue")
        request, charge_tokens = selection
        lane = self._queue_index.get(request.request_id, self._waiting)
        lane.remove_request(request)
        if charge_tokens > 0:
            self._long_bucket -= charge_tokens
        self._forget(request, lane)
        self._selection = None
        return request

    def peek_request(self) -> Request:
        selection = self._ensure_selection()
        if selection is None:
            raise IndexError("peek from an empty LengthAwareRequestQueue")
        return selection[0]

    def prepend_request(self, request: Request) -> None:
        # Prepend == recovery/resume; always route to the immediate lane so the
        # request is dispatched before any newly arrived waiting prefill.
        self._force_immediate_ids.add(request.request_id)
        self._immediate.prepend_request(request)
        self._track(request, self._immediate)
        self._selection = None

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        lane = self._queue_index.get(request.request_id)
        if lane is None:
            raise ValueError("request not found in LengthAwareRequestQueue")
        lane.remove_request(request)
        self._forget(request, lane)
        self._selection = None

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            lane = self._queue_index.get(request.request_id)
            if lane is None:
                continue
            lane.remove_request(request)
            self._forget(request, lane)
        self._selection = None

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._immediate) + len(self._waiting)

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate
        yield from self._waiting

    def __contains__(self, request: object) -> bool:
        rid = getattr(request, "request_id", None)
        return rid is not None and rid in self._queue_index


def create_request_queue(
    policy: SchedulingPolicy,
    *,
    length_fn: Callable[[Request], int] | None = None,
    threshold: int = 256,
    long_max_wait_ms: float = 0.0,
    long_token_reservation: float = 0.0,
    long_burst_steps: int = 4,
    immediate_predicate: Callable[[Request], bool] | None = None,
) -> RequestQueue:
    """Create request queue based on scheduling policy.

    The keyword-only arguments are only consumed by
    ``SchedulingPolicy.LENGTH_AWARE`` and ignored otherwise.
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.LENGTH_AWARE:
        return LengthAwareRequestQueue(
            length_fn=length_fn,
            threshold=threshold,
            long_max_wait_ms=long_max_wait_ms,
            long_token_reservation=long_token_reservation,
            long_burst_steps=long_burst_steps,
            immediate_predicate=immediate_predicate,
        )
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
