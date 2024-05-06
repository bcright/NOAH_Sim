class Config:
    # 通用配置
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

    # 监视器配置
    MONITOR_SAMPLE_RATE = 5  # 监视器数据采样率（秒）

    # 限制器配置
    REQUEST_THRESHOLD = 100  # 每分钟允许的最大请求数

    # DRL模型配置
    DRL_MODEL_PATH = 'path/to/your/model'  # 模型文件路径

# 开发环境特有配置
class DevelopmentConfig(Config):
    DEBUG = True

# 测试环境特有配置
class TestingConfig(Config):
    DEBUG = True
    TESTING = True

# 生产环境特有配置
class ProductionConfig(Config):
    DEBUG = False
    REQUEST_THRESHOLD = 150  # 生产环境可能需要更高的请求阈值