def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running Monte Carlo (full N); deselect with -m 'not slow'"
    )
