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

    Inspired by vllm-ascend PR #10285 (LAPS). The waiting queue is split into
    three lanes and drained with priority ``immediate > aged-long > short >
    long``:

    - **immediate**: recovery-style requests (preempted / already partially
      computed) that should be resumed first. Populated via ``prepend_request``
      or the optional ``immediate_predicate``.
    - **short**: prefills whose number of tokens *to compute* is
      ``<= threshold``.
    - **long**: prefills whose number of tokens *to compute* is ``> threshold``.

    The key difference from LAPS: the short/long split uses ``length_fn(request)``
    -- the number of tokens that still need to be computed **after prefix-cache
    hits are removed** -- instead of the raw prompt length. So a long prompt that
    is mostly a cache hit is (correctly) treated as short.

    Anti-starvation uses a token bucket: it refills by
    ``long_token_reservation * per-step token budget`` every ``begin_step`` and,
    once the oldest long has waited ``>= long_max_wait_ms``, an aged long may be
    promoted ahead of waiting shorts only while the bucket has credit. Admitting
    an aged long ahead of shorts spends its full remaining compute length from
    the bucket, so the aged-long admission rate stays bounded by the reservation
    fraction and shorts always retain ``1 - reservation`` of throughput.

    The three internal lanes are plain FCFS queues so that "oldest long" and
    arrival order are well defined; this policy therefore assumes a single total
    arrival order (like FCFS).

    NOTE: the bucket is charged inside ``pop_request``. The scheduler also pops
    from this queue on rare "skip" paths (e.g. a request that momentarily
    exceeds ``max_loras``); those pops may count as an aged-long promotion. This
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

        # Inner lanes are FCFS so arrival order (and thus "oldest long") holds.
        self._immediate = FCFSRequestQueue()
        self._short = FCFSRequestQueue()
        self._long = FCFSRequestQueue()

        # req_id -> the lane currently holding the request.
        self._queue_index: dict[str, RequestQueue] = {}
        # req_id -> monotonic timestamp the request entered the long lane.
        self._long_enqueue_at: dict[str, float] = {}
        # req_id -> tokens-to-compute recorded at classification (bucket charge).
        self._compute_len: dict[str, int] = {}
        # req_ids explicitly forced into the immediate lane (recovery flows).
        self._force_immediate_ids: set[str] = set()

        # Token bucket throttling aged-long admissions (see begin_step).
        self._long_bucket = 0.0
        self._long_bucket_capacity = 0.0
        self._long_refill_per_step = 0.0

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

    # ------------------------------------------------------------------ #
    # Classification & bookkeeping
    # ------------------------------------------------------------------ #
    def _classify(self, request: Request, force_immediate: bool) -> RequestQueue:
        rid = request.request_id
        if force_immediate or rid in self._force_immediate_ids:
            return self._immediate
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate
        compute_len = max(self.length_fn(request), 0)
        self._compute_len[rid] = compute_len
        return self._short if compute_len <= self.threshold else self._long

    def _track(self, request: Request, queue: RequestQueue) -> None:
        rid = request.request_id
        self._queue_index[rid] = queue
        if queue is self._long:
            self._long_enqueue_at.setdefault(rid, time.monotonic())

    def _forget(self, request: Request, queue: RequestQueue) -> None:
        rid = request.request_id
        self._queue_index.pop(rid, None)
        self._long_enqueue_at.pop(rid, None)
        self._compute_len.pop(rid, None)
        self._force_immediate_ids.discard(rid)

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def _long_head_wait_ms(self) -> float | None:
        if self.long_max_wait_ms <= 0 or not self._long:
            return None
        head = self._long.peek_request()  # FCFS head = oldest long.
        enqueued_at = self._long_enqueue_at.get(head.request_id)
        if enqueued_at is None:
            return None
        return (time.monotonic() - enqueued_at) * 1000.0

    def _select(self) -> RequestQueue | None:
        """Pick the lane to dispatch from (no side effects)."""
        if self._immediate:
            return self._immediate
        if self._long:
            wait_ms = self._long_head_wait_ms()
            if wait_ms is not None and wait_ms >= self.long_max_wait_ms:
                if not self._short:
                    # Stall avoidance: no short is waiting, so run the long.
                    return self._long
                if self._long_bucket > 0.0:
                    # Reservation credit available: promote the aged long.
                    return self._long
                # Bucket exhausted and shorts are waiting: prefer shorts so the
                # aged-long admission rate stays bounded by the reservation.
        if self._short:
            return self._short
        if self._long:
            return self._long
        return None

    def _maybe_charge_aged_long(
        self, request: Request, queue: RequestQueue
    ) -> None:
        if queue is not self._long or self.long_max_wait_ms <= 0:
            return
        enqueued_at = self._long_enqueue_at.get(request.request_id)
        if enqueued_at is None:
            return
        wait_ms = (time.monotonic() - enqueued_at) * 1000.0
        if wait_ms < self.long_max_wait_ms:
            return
        # Only counts as a promotion (and a charge) if it actually jumped waiting
        # shorts. When no short is waiting it is stall avoidance -- not charged.
        if len(self._short) == 0:
            return
        self._long_bucket -= self._compute_len.get(request.request_id, 0)

    # ------------------------------------------------------------------ #
    # RequestQueue interface
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request) -> None:
        queue = self._classify(request, force_immediate=False)
        queue.add_request(request)
        self._track(request, queue)

    def pop_request(self) -> Request:
        queue = self._select()
        if queue is None:
            raise IndexError("pop from an empty LengthAwareRequestQueue")
        request = queue.pop_request()
        self._maybe_charge_aged_long(request, queue)
        self._forget(request, queue)
        return request

    def peek_request(self) -> Request:
        queue = self._select()
        if queue is None:
            raise IndexError("peek from an empty LengthAwareRequestQueue")
        return queue.peek_request()

    def prepend_request(self, request: Request) -> None:
        # Prepend == recovery/resume; always route to the immediate lane so the
        # request is dispatched before any newly arrived short/long prefill.
        self._force_immediate_ids.add(request.request_id)
        self._immediate.prepend_request(request)
        self._track(request, self._immediate)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        queue = self._queue_index.get(request.request_id)
        if queue is None:
            raise ValueError("request not found in LengthAwareRequestQueue")
        queue.remove_request(request)
        self._forget(request, queue)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            queue = self._queue_index.get(request.request_id)
            if queue is None:
                continue
            queue.remove_request(request)
            self._forget(request, queue)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._immediate) + len(self._short) + len(self._long)

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate
        yield from self._short
        yield from self._long

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
