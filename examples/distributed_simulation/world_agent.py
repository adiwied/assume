# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast

from .config import (
    agent_adress,
    index,
    market_operator_addr,
    marketdesign,
    worker,
)


async def create_worker(
    world: World,
    marketdesign: list[MarketConfig],
    i: int,
    n_proc: int,
    m_agents: int = 1,
):
    for market_config in marketdesign:
        market_config.addr = market_operator_addr
        world.markets[f"{market_config.market_id}"] = market_config

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    for m in range(m_agents):
        world.add_unit_operator(f"my_operator{i} {m}")
        world.add_unit(
            f"nuclear{i} {m}",
            "power_plant",
            f"my_operator{i} {m}",
            {
                "min_power": 200 / (n_proc * m_agents),
                "max_power": 1000 / (n_proc * m_agents),
                "bidding_strategies": {"EOM": "naive_eom"},
                "technology": "nuclear",
            },
            nuclear_forecast,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        i = int(sys.argv[1])
        n = int(sys.argv[2])
        agent_adress = ("0.0.0.0", int(sys.argv[3]))
    else:
        i = 0
        n = 1
    world = World(addr=agent_adress, distributed_role=False)
    world.loop.run_until_complete(worker(world, marketdesign, create_worker, i, n))
