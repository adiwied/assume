# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import NaiveStrategy
from assume.units.demand import Demand


def test_demand():
    strategies = {"energy": NaiveStrategy()}

    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    product_tuple = (
        datetime(2023, 7, 1, hour=1),
        datetime(2023, 7, 1, hour=2),
        None,
    )
    ff = NaiveForecast(index, demand=150)
    dem = Demand(
        "demand_test01",
        "UO1",
        "energy",
        strategies,
        index,
        150,
        0,
        forecaster=ff,
        price=2000,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    min_power, max_power = dem.calculate_min_max_power(start, end)

    assert max_power.max() == -150
    assert dem.calculate_marginal_cost(start, max_power.max()) == 2000

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "start_time" in bid.keys()
        assert "end_time" in bid.keys()
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1


def test_demand_series():
    strategies = {"energy": NaiveStrategy()}

    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    product_tuple = (
        datetime(2023, 7, 1, hour=1),
        datetime(2023, 7, 1, hour=2),
        None,
    )

    demand = pd.Series(100, index=index)
    demand.iloc[1] = 80
    price = pd.Series(1000, index=index)
    price.iloc[1] = 0

    ff = NaiveForecast(index, demand=demand)
    dem = Demand(
        "demand_test02",
        "UO1",
        "energy",
        strategies,
        index,
        150,
        0,
        forecaster=ff,
        price=price,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    min_power, max_power = dem.calculate_min_max_power(start, end)

    # power should be the highest demand which is used throughout the period
    # in our case 80 MW
    max_power = max_power.max()
    assert max_power == -80

    assert dem.calculate_marginal_cost(start, max_power) == 0
    assert dem.calculate_marginal_cost(end, max_power) == 1000

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1
    assert bids[0]["volume"] == max_power
    assert bids[0]["price"] == price.iloc[1]


if __name__ == "__main__":
    test_demand()
