import numpy as np
import simpy


RPS = 30


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
        self.api = simpy.Resource(env, api_count)
        self.web = simpy.Resource(env, web_count)
        self.res_manage = simpy.Resource(env, res_manage_count)
        self.cus_manage = simpy.Resource(env, cus_manage_count)
        self.ord_manage = simpy.Resource(env, ord_manage_count)
        self.del_relation = simpy.Resource(env, del_relation_count)
        self.payment = simpy.Resource(env, payment_count)

    def api_service(self, request, rate=2):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def web_service(self, request, rate=3):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def res_manage_service(self, request, rate=8):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def cus_manage_service(self, request, rate=5):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def ord_manage_service(self, request, rate=6):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def del_relation_service(self, request, rate=9):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )

    def payment_service(self, request, rate=12):
        yield self.env.timeout(
            np.random.exponential(1/rate)
        )


def register_order_mobile(env: simpy.Environment, request, snapp_food: SnappFood):

    with snapp_food.payment.request() as req:
        yield req
        yield env.process(snapp_food.payment_service(request))

    with snapp_food.ord_manage.request() as req:
        yield req
        yield env.process(snapp_food.ord_manage_service(request))

    with snapp_food.api.request() as req:
        yield req
        yield env.process(snapp_food.api_service(request))


def register_order_web(env: simpy.Environment, request, snapp_food: SnappFood):

    with snapp_food.payment.request() as req:
        yield req
        yield env.process(snapp_food.payment_service(request))

    with snapp_food.ord_manage.request() as req:
        yield req
        yield env.process(snapp_food.ord_manage_service(request))

    with snapp_food.web.request() as req:
        yield req
        yield env.process(snapp_food.web_service(request))


def send_message_to_del(env: simpy.Environment, request, snapp_food: SnappFood):

    with snapp_food.del_relation.request() as req:
        yield req
        yield env.process(snapp_food.del_relation_service(request))

    with snapp_food.cus_manage.request() as req:
        yield req
        yield env.process(snapp_food.cus_manage_service(request))

    with snapp_food.api.request() as req:
        yield req
        yield env.process(snapp_food.api_service(request))


def view_res_info_mobile(env, request, snapp_food: SnappFood):

    with snapp_food.res_manage.request() as req:
        yield req
        yield env.process(snapp_food.res_manage_service(request))

    with snapp_food.api.request() as req:
        yield req
        yield env.process(snapp_food.api_service(request))


def view_res_info_web(env, request, snapp_food: SnappFood):

    with snapp_food.res_manage.request() as req:
        yield req
        yield env.process(snapp_food.res_manage_service(request))

    with snapp_food.web.request() as req:
        yield req
        yield env.process(snapp_food.web_service(request))


def request_del(env, request, snapp_food: SnappFood):

    with snapp_food.del_relation.request() as req:
        yield req
        yield env.process(snapp_food.del_relation_service(request))

    with snapp_food.res_manage.request() as req:
        yield req
        yield env.process(snapp_food.res_manage_service(request))

    with snapp_food.web.request() as req:
        yield req
        yield env.process(snapp_food.web_service(request))


def follow_up_ord(env, request, snapp_food: SnappFood):

    with snapp_food.ord_manage.request() as req:
        yield req
        yield env.process(snapp_food.ord_manage_service(request))

    with snapp_food.api.request() as req:
        yield req
        yield env.process(snapp_food.api_service(request))


def run_snapp_food(
        env: simpy.Environment,
        api_count: int,
        web_count: int,
        res_manage_count: int,
        cus_manage_count: int,
        ord_manage_count: int,
        del_relation_count: int,
        payment_count: int,
):
    snapp_food = SnappFood(
        env, api_count, web_count,
        res_manage_count, cus_manage_count,
        ord_manage_count, del_relation_count,
        payment_count
    )
    request = 0

    while True:
        yield env.timeout(1)

        for i in range(RPS):
            kind = np.random.uniform()
            if 0 <= kind < 0.2:
                env.process(register_order_mobile(env, request, snapp_food))
            elif 0.2 <= kind < 0.3:
                env.process(register_order_web(env, request, snapp_food))
            elif 0.3 <= kind < 0.35:
                env.process(send_message_to_del(env, request, snapp_food))
            elif 0.35 <= kind < 0.6:
                env.process(view_res_info_mobile(env, request, snapp_food))
            elif 0.6 <= kind < 0.75:
                env.process(view_res_info_web(env, request, snapp_food))
            elif 0.75 <= kind < 0.95:
                env.process(request_del(env, request, snapp_food))
            else:
                env.process(follow_up_ord(env, request, snapp_food))


def main():
    env = simpy.Environment()

    api_count = 1
    web_count = 1
    res_manage_count = 1
    cus_manage_count = 1
    ord_manage_count = 1
    del_relation_count = 1
    payment_count = 1

    env.process(run_snapp_food(
        env, api_count, web_count,
        res_manage_count, cus_manage_count,
        ord_manage_count, del_relation_count,
        payment_count
    ))

    env.run(until=3600)


if __name__ == '__main__':
    main()
