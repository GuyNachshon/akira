import pytest
from network_defense_system.environment import NetworkEnvironment

def test_environment_initialization():
    env = NetworkEnvironment(num_nodes=10)
    assert env.num_nodes == 10