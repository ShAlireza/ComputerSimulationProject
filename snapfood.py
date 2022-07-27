from functools import partial, wraps

import numpy as np
import matplotlib.pyplot as plt
import simpy
from types import FunctionType

RPS = 30
TOTAL_TIME = 90
TIME_DIVIDER = 60

requests_usage_times = {}
requests_total_times = {}
requests = {}


class Request:
    def __init__(self, number, timeout):
        self.number = number
        self.timeout = timeout
        self.error = False


def methods(cls):
    return [x for x, y in cls.__dict__.items()
            if type(y) == FunctionType and not x.startswith('_')]


class VerboseResource(simpy.PriorityResource):

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name')
        kwargs.pop('name')
        super().__init__(*args, **kwargs)


def patch_resource(resource, pre=None, post=None):

    def get_wrapper(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if pre:
                pre(resource)

            ret = func(*args, **kwargs)

            if post:
                post(resource)

            return ret
        return wrapper

    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))


def monitor(data, resource):

    item = (
        resource._env.now,
        resource.count,
        len(resource.queue),
    )

    data.append(item)


class SnappFood:
    def __init__(
        self,
        env: simpy.Environment,
        api_count: int,
        web_count: int,
        res_manage_count: int,
        cus_manage_count: int,
        ord_manage_count: int,
        del_relation_count: int,
        payment_count: int,
    ):
        self.env = env
        self.api_data = []
        self.api = self._make_monitoring_resource(
            VerboseResource(env, api_count, name='api'), self.api_data)

        self.web_data = []
        self.web = self._make_monitoring_resource(
            VerboseResource(env, web_count, name='web'), self.web_data)

        self.res_manage_data = []
        self.res_manage = self._make_monitoring_resource(
            VerboseResource(env, res_manage_count, name='res_manage'), self.res_manage_data)

        self.cus_manage_data = []
        self.cus_manage = self._make_monitoring_resource(
            VerboseResource(env, cus_manage_count, name='cus_manage'), self.cus_manage_data)

        self.ord_manage_data = []
        self.ord_manage = self._make_monitoring_resource(
            VerboseResource(env, ord_manage_count, name='ord_manage'), self.ord_manage_data)

        self.del_relation_data = []
        self.del_relation = self._make_monitoring_resource(
            VerboseResource(env, del_relation_count, name='del_relation'), self.del_relation_data)

        self.payment_data = []
        self.payment = self._make_monitoring_resource(
            VerboseResource(env, payment_count, name='payment'), self.payment_data)

        self.data = [self.api_data, self.web_data, self.res_manage_data,
                     self.cus_manage_data, self.ord_manage_data,
                     self.del_relation_data, self.payment_data]

    def _make_monitoring_resource(self, resource, data):
        monitor_res = partial(monitor, data)
        patch_resource(resource, post=monitor_res)

        return resource

    def api_service(self, request, rate=2, error_rate=0.01):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def web_service(self, request, rate=3, error_rate=0.01):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def res_manage_service(self, request, rate=8, error_rate=0.02):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def cus_manage_service(self, request, rate=5, error_rate=0.02):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def ord_manage_service(self, request, rate=6, error_rate=0.03):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def del_relation_service(self, request, rate=9, error_rate=0.1):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)

    def payment_service(self, request, rate=12, error_rate=0.2):
        time = np.ceil(np.random.exponential(1/rate))

        for t in range(time):
            if np.random.uniform() <= error_rate:
                request.error = True
                return request
            yield self.env.timeout(1 / TIME_DIVIDER)


def init():
    global requests_usage_times
    global requests_total_times
    global requests

    requests_usage_times = {}
    requests_total_times = {}
    requests = {}
    for x in methods(SnappFood) + ['register_order_mobile',
                                   'register_order_web',
                                   'send_message_to_del',
                                   'view_res_info_mobile',
                                   'view_res_info_web',
                                   'request_del',
                                   'follow_up_ord'
                                   ]:
        requests_total_times[x] = []
        requests_usage_times[x] = []
        requests[x] = []


def register_order_mobile(env: simpy.Environment, request, snapp_food: SnappFood):

    request_start_time = env.now
    start_time = env.now

    requests['register_order_mobile'].append(request)

    with snapp_food.payment.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.payment_service(request))

        if request.error:
            return

        requests_usage_times['payment_service'].append(
            env.now - service_start_time)
    requests_total_times['payment_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.ord_manage.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.ord_manage_service(request))

        if request.error:
            return

        requests_usage_times['ord_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['ord_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.api.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.api_service(request))

        if request.error:
            return

        requests_usage_times['api_service'].append(
            env.now - service_start_time)
    requests_total_times['api_service'].append(env.now - start_time)

    requests_total_times['register_order_mobile'].append(
        env.now - request_start_time
    )


def register_order_web(env: simpy.Environment, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['register_order_web'].append(request)

    with snapp_food.payment.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.payment_service(request))

        if request.error:
            return

        requests_usage_times['payment_service'].append(
            env.now - service_start_time)
    requests_total_times['payment_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.ord_manage.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.ord_manage_service(request))

        if request.error:
            return

        requests_usage_times['ord_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['ord_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.web.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.web_service(request))

        if request.error:
            return

        requests_usage_times['web_service'].append(
            env.now - service_start_time)
    requests_total_times['web_service'].append(env.now - start_time)

    requests_total_times['register_order_web'].append(
        env.now - request_start_time
    )


def send_message_to_del(env: simpy.Environment, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['send_message_to_del'].append(request)
    with snapp_food.del_relation.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.del_relation_service(request))

        if request.error:
            return

        requests_usage_times['del_relation_service'].append(
            env.now - service_start_time)
    requests_total_times['del_relation_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.cus_manage.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.cus_manage_service(request))

        if request.error:
            return

        requests_usage_times['cus_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['cus_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.api.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.api_service(request))

        if request.error:
            return

        requests_usage_times['api_service'].append(
            env.now - service_start_time)
    requests_total_times['api_service'].append(env.now - start_time)

    requests_total_times['send_message_to_del'].append(
        env.now - request_start_time
    )


def view_res_info_mobile(env, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['view_res_info_mobile'].append(request)
    with snapp_food.res_manage.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.res_manage_service(request))

        if request.error:
            return

        requests_usage_times['res_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['res_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.api.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.api_service(request))

        if request.error:
            return

        requests_usage_times['api_service'].append(
            env.now - service_start_time)
    requests_total_times['api_service'].append(env.now - start_time)

    requests_total_times['view_res_info_mobile'].append(
        env.now - request_start_time
    )


def view_res_info_web(env, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['view_res_info_web'].append(request)
    with snapp_food.res_manage.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.res_manage_service(request))

        if request.error:
            return

        requests_usage_times['res_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['res_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.web.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.web_service(request))

        if request.error:
            return

        requests_usage_times['web_service'].append(
            env.now - service_start_time)
    requests_total_times['web_service'].append(env.now - start_time)

    requests_total_times['view_res_info_web'].append(
        env.now - request_start_time
    )


def request_del(env, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['request_del'].append(request)
    with snapp_food.del_relation.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.del_relation_service(request))

        if request.error:
            return

        requests_usage_times['del_relation_service'].append(
            env.now - service_start_time)
    requests_total_times['del_relation_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.res_manage.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.res_manage_service(request))

        if request.error:
            return

        requests_usage_times['res_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['res_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.web.request(priority=1, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.web_service(request))

        if request.error:
            return

        requests_usage_times['web_service'].append(
            env.now - service_start_time)
    requests_total_times['web_service'].append(env.now - start_time)

    requests_total_times['request_del'].append(
        env.now - request_start_time
    )


def follow_up_ord(env, request, snapp_food: SnappFood):
    request_start_time = env.now
    start_time = env.now
    requests['follow_up_ord'].append(request)
    with snapp_food.ord_manage.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.ord_manage_service(request))

        if request.error:
            return

        requests_usage_times['ord_manage_service'].append(
            env.now - service_start_time)
    requests_total_times['ord_manage_service'].append(env.now - start_time)

    start_time = env.now
    with snapp_food.api.request(priority=2, preempt=False) as req:
        yield req
        service_start_time = env.now
        yield env.process(snapp_food.api_service(request))

        if request.error:
            return

        requests_usage_times['api_service'].append(
            env.now - service_start_time)
    requests_total_times['api_service'].append(env.now - start_time)

    requests_total_times['follow_up_ord'].append(
        env.now - request_start_time
    )


def run_snapp_food(
        env: simpy.Environment,
        snapp_food: SnappFood,
        register_order_mobile_to=10**10,
        register_order_web_to=10**10,
        send_message_to_del_to=10**10,
        view_res_info_mobile_to=10**10,
        view_res_info_web_to=10**10,
        request_del_to=10**10,
        follow_up_ord_to=10**10
):
    request_num = 0

    while True:
        for i in range(RPS):
            kind = np.random.uniform()
            if 0 <= kind < 0.2:
                env.process(
                    register_order_mobile(
                        env,
                        Request(number=request_num + i,
                                timeout=register_order_mobile_to),
                        snapp_food
                    )
                )
            elif 0.2 <= kind < 0.3:
                env.process(
                    register_order_web(
                        env,
                        Request(number=request_num + i,
                                timeout=register_order_web_to),
                        snapp_food
                    )
                )
            elif 0.3 <= kind < 0.35:
                env.process(
                    send_message_to_del(
                        env,
                        Request(number=request_num + i,
                                timeout=send_message_to_del_to),
                        snapp_food
                    )
                )
            elif 0.35 <= kind < 0.6:
                env.process(
                    view_res_info_mobile(
                        env,
                        Request(number=request_num + i,
                                timeout=view_res_info_mobile_to),
                        snapp_food
                    )
                )
            elif 0.6 <= kind < 0.75:
                env.process(
                    view_res_info_web(
                        env,
                        Request(number=request_num + i,
                                timeout=view_res_info_web_to),
                        snapp_food
                    )
                )
            elif 0.75 <= kind < 0.95:
                env.process(
                    request_del(
                        env,
                        Request(number=request_num + i,
                                timeout=request_del_to),
                        snapp_food
                    )
                )
            else:
                env.process(
                    follow_up_ord(
                        env,
                        Request(number=request_num + i,
                                timeout=follow_up_ord_to),
                        snapp_food
                    )
                )

        yield env.timeout(1 / TIME_DIVIDER)
        request_num += RPS


def main(
    rps: int = 30,
    simulation_time: int = 90,
    api_count: int = 1,
    web_count: int = 1,
    res_manage_count: int = 1,
    cus_manage_count: int = 1,
    ord_manage_count: int = 1,
    del_relation_count: int = 1,
    payment_count: int = 1,
):

    init()

    global TOTAL_TIME
    TOTAL_TIME = simulation_time / TIME_DIVIDER

    global RPS
    RPS = rps

    env = simpy.Environment()

    snapp_food = SnappFood(
        env, api_count, web_count,
        res_manage_count, cus_manage_count,
        ord_manage_count, del_relation_count,
        payment_count
    )

    env.process(run_snapp_food(
        env, snapp_food
    ))

    env.run(until=TOTAL_TIME)

    return snapp_food


def Q1(snapp_food: SnappFood):
    for service_data in snapp_food.data:
        arr = np.array(service_data)
        # x = arr[:, [0]]
        # y = arr[:, [2]]
        # Plot lines with different marker sizes
        # plt.plot(x, y, marker='s', ms=5, linewidth=1, color='darkorange') # square

        # plt.ylabel('queue length',fontsize=12)

        # plt.xlabel('time',fontsize=12)
        # plt.yticks(fontsize=11)

        # plt.xticks(range(0, TOTAL_TIME, 10), fontsize=12)
        # plt.show()

        print(f'Average queue size: {np.average(arr[:, [2]])}')


def Q2(snpp_food: SnappFood):
    print('Services:')
    for ind, x in enumerate(requests_total_times.items()):
        if ind < 7:
            print(f'average queue time {x[0]}: \t {np.average(x[1]):.2f}')

    print('Requests:')
    for ind, x in enumerate(requests_total_times.items()):
        if ind >= 7:
            print(f'average queue time {x[0]}: \t {np.average(x[1]):.2f}')


def Q3(snapp_food: SnappFood):
    for x, y in requests_usage_times.items():
        y = np.array(y)
        utilization = np.sum(y) / TOTAL_TIME
        if utilization > 0.001:
            print(utilization)


def answer_all_qeustions(snapp_food: SnappFood):
    Q1(snapp_food)
    Q2(snapp_food)
    Q3(snapp_food)
