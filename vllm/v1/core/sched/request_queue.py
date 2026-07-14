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
        """调度器在每个调度步开始、开始消费等待队列之前调用一次的钩子。

        有状态的策略（例如长度感知的令牌桶 aging）用它来刷新每步状态。默认是空操作，
        因此 FCFS/priority 队列不受影响。
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
    """兜底的计算长度：prompt/decode 长度减去已计算的 token 数。当没有提供
    感知前缀缓存的 ``length_fn`` 时使用。"""
    return max(request.num_tokens - request.num_computed_tokens, 0)


class LengthAwareRequestQueue(RequestQueue):
    """长度感知的 prefill 调度队列。

    灵感来自 vllm-ascend PR #10285 (LAPS)，但有一个关键的结构性差异：
    **请求不在入队时被划分为短/长。** 一旦涉及缓存，prompt 长度就不能很好地代表开销；
    更重要的是，实际还需计算多少 token 取决于*实时*的缓存状态（本地前缀缓存 + 外部
    KV 连接器缓存），它随时间变化，且只有在调度时才能廉价且安全地得知。

    因此分类被推迟到**选择时**惰性进行：每当一个请求即将被准入 running 集合时，
    队列会遍历所有等待请求，通过 ``length_fn(request)`` 实时计算每个请求的真实计算
    长度（prompt 减去本地*和*远端缓存命中），并据此划分为短（``compute_len <=
    threshold``）与长（``> threshold``）。派发优先级为::

        immediate > aged-long > short > long

    - **immediate**：恢复类请求（被抢占 / 已部分计算），最先派发。通过
      ``prepend_request`` 或可选的 ``immediate_predicate`` 填充。它们放在一条独立的
      小队列里，不参与长度扫描。
    - **short**：实时计算长度 ``<= threshold`` 的、最老的等待请求。
    - **long**：其余请求，仅当没有短请求可派发、或最老的长请求已 aging 超过
      ``long_max_wait_ms`` 时才派发。

    防饿死用令牌桶：每次 ``begin_step`` 回填 ``long_token_reservation * 每步 token
    预算``；当最老的长请求已等待 ``>= long_max_wait_ms`` 时，只有在桶还有额度的情况下
    该长请求才能被晋升到等待短请求之前。让一个 aged-long 越过短请求会从桶里扣掉它的
    整份剩余计算长度，因此 aged-long 的准入速率被限制在预留比例内，短请求始终保留
    ``1 - reservation`` 的吞吐。

    除 immediate 外的请求放在单一的 FCFS 结构里，从而“最老”是明确定义的；该策略假定
    存在单一的全序到达顺序（与 FCFS 相同）。

    开销说明：选择的复杂度为 O(N)（N 为等待请求数，会对每个请求求值 ``length_fn``），
    因此每准入一个请求就要扫描一次。``peek_request`` 的结果会被缓存，供紧随其后的
    ``pop_request`` 复用（两者之间不会有队列变更），并在任何变更以及 ``begin_step``
    时失效。

    注意：扣桶发生在 ``pop_request`` 内。调度器在少数“跳过”路径上也会从本队列 pop
    （例如某请求暂时超出 ``max_loras``），这样的 pop 可能被计为一次 aged-long 晋升。
    这是有意为之的良性近似——偶尔多扣一次只会让 aged-long 准入更慢一点，方向上偏向
    保护短请求。
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

        # 恢复/被抢占的请求，最先派发（不参与长度扫描）。
        self._immediate = FCFSRequestQueue()
        # 其余所有请求，按到达顺序存放。这里不做短/长划分——短/长在选择时根据实时
        # 计算长度决定。
        self._waiting = FCFSRequestQueue()

        # req_id -> 当前持有该请求的 lane。
        self._queue_index: dict[str, RequestQueue] = {}
        # req_id -> 该请求进入等待 lane 的单调时间戳（用于 aging）。
        self._enqueue_at: dict[str, float] = {}
        # 被显式强制放入 immediate lane 的 req_id（恢复流程）。
        self._force_immediate_ids: set[str] = set()

        # 限制 aged-long 准入速率的令牌桶（见 begin_step）。
        self._long_bucket = 0.0
        self._long_bucket_capacity = 0.0
        self._long_refill_per_step = 0.0

        # 缓存的选择结果：(request, charge_tokens)。仅当是“越过等待短请求的 aged-long
        # 晋升”时 charge_tokens > 0。在任何变更以及 begin_step 时失效。
        self._selection: tuple[Request, int] | None = None

    # ------------------------------------------------------------------ #
    # 每步钩子
    # ------------------------------------------------------------------ #
    def begin_step(self, token_budget: int) -> None:
        """为本调度步回填 aged-long 的准入令牌桶。"""
        self._long_refill_per_step = self.long_token_reservation * token_budget
        self._long_bucket_capacity = self._long_refill_per_step * self.long_burst_steps
        self._long_bucket = min(
            self._long_bucket_capacity,
            self._long_bucket + self._long_refill_per_step,
        )
        # 距上一步，实时缓存状态可能已改变，因此任何依赖计算长度的缓存选择结果现在
        # 都已过时。
        self._selection = None

    # ------------------------------------------------------------------ #
    # 记账
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
    # 选择（每次都遍历所有等待请求）
    # ------------------------------------------------------------------ #
    def _compute_selection(self) -> tuple[Request, int] | None:
        # immediate（恢复类）请求总是优先，且不参与长度扫描。
        if self._immediate:
            return (self._immediate.peek_request(), 0)

        now = time.monotonic()
        oldest_short: Request | None = None
        oldest_long: Request | None = None
        oldest_long_len = 0
        has_short = False

        # 按到达顺序单趟遍历：遇到的第一个短/长即为最老的短/长。对每个请求都实时
        # 求值其计算长度。
        for request in self._waiting:
            compute_len = max(self.length_fn(request), 0)
            if compute_len <= self.threshold:
                has_short = True
                if oldest_short is None:
                    oldest_short = request
            elif oldest_long is None:
                oldest_long = request
                oldest_long_len = compute_len

        # 防饿死：最老的长请求一旦 aging，可在令牌桶额度允许下被晋升到等待短请求之前。
        if oldest_long is not None and self.long_max_wait_ms > 0:
            enqueued_at = self._enqueue_at.get(oldest_long.request_id, now)
            waited_ms = (now - enqueued_at) * 1000.0
            if waited_ms >= self.long_max_wait_ms:
                if not has_short:
                    # 避免空转：没有短请求在等，直接跑这个长请求（不扣桶）。
                    return (oldest_long, 0)
                if self._long_bucket > 0.0:
                    # 桶还有额度：晋升该长请求，并扣掉它的整份剩余计算长度。
                    return (oldest_long, oldest_long_len)
                # 桶已耗尽且有短请求在等：优先短请求，从而把 aged-long 的准入速率
                # 限制在预留比例之内。

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
    # RequestQueue 接口
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
        # prepend == 恢复/续跑；一律放入 immediate lane，使该请求先于任何新到达的
        # 等待 prefill 被派发。
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
    """根据调度策略创建请求队列。

    仅 ``SchedulingPolicy.LENGTH_AWARE`` 会使用这些仅限关键字参数，其它策略忽略之。
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
