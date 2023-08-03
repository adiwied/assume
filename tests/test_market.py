import calendar
from datetime import datetime

import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping

from assume.markets.base_market import MarketConfig, MarketProduct, MarketRole

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)


@pytest.fixture
async def market_role() -> MarketRole:
    market_name = "Test"
    marketconfig = MarketConfig(
        market_name,
        rr.rrule(rr.HOURLY, dtstart=start, until=end),
        rd(hours=1),
        "pay_as_clear",
        [MarketProduct(rd(hours=1), 1, rd(hours=1))],
    )
    clock = ExternalClock(0)
    container = await create_container(addr=("0.0.0.0", 9098), clock=clock)
    market_agent = RoleAgent(container)
    market_role = MarketRole(marketconfig=marketconfig)
    market_agent.add_role(market_role)

    yield market_role

    end_ts = calendar.timegm(end.utctimetuple())
    clock.set_time(end_ts)
    await tasks_complete_or_sleeping(container)
    await container.shutdown()


async def test_market_init(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }

    orderbook = [
        {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]

    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1


async def test_market_tick(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    market_role.marketconfig.price_tick = 0.1

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120.123,
            "price": 1201,  # used to present 120.1 with precision 0.1
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1
    assert market_role.all_orders[0]["price"] == 1201

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120.123,
            "price": 120.123,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]

    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # this does not work
    assert len(market_role.all_orders) == 1


async def test_market_max(market_role: MarketRole):
    meta = {
        "sender_addr": market_role.context.addr,
        "sender_id": market_role.context.aid,
    }
    market_role.marketconfig.maximum_bid_price = 1000
    market_role.marketconfig.minimum_bid_price = -500
    market_role.marketconfig.maximum_bid_volume = 9090

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9091,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # volume is too high
    assert len(market_role.all_orders) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": 1001,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # price is too high
    assert len(market_role.all_orders) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": -550,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    # price is too low
    assert len(market_role.all_orders) == 0

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 9090,
            "price": 1000,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)
    assert len(market_role.all_orders) == 1
    assert market_role.all_orders[0]["price"] == 1000
    assert market_role.all_orders[0]["volume"] == 9090


async def test_market_registration(market_role: MarketRole):
    meta = {
        "sender_addr": "test_address",
        "sender_id": "test_aid",
    }

    assert market_role.registered_agents == []
    market_role.handle_registration({}, meta=meta)
    assert len(market_role.registered_agents) == 1
    assert market_role.registered_agents[0] == tuple(meta.values())


async def test_market_unmatched(market_role: MarketRole):
    meta = {
        "sender_addr": "test_address",
        "sender_id": "test_aid",
    }

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 20,
            "agent_id": "gen1",
            "only_hours": None,
        }
    ]
    market_role.handle_orderbook(content={"orderbook": orderbook}, meta=meta)

    content = {
        "order": {
            "start_time": start,
            "end_time": start + rd(hours=1),
            "only_hours": None,
        }
    }

    market_role.handle_get_unmatched(content, meta)