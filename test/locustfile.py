from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    # 用户等待时间定义为1到3秒之间
    wait_time = between(1, 3)

    @task
    def load_test(self):
        # 发送GET请求
        self.client.get("/request")

    @task(3)  # 这个任务的权重是前一个的3倍，即执行频率更高
    def load_test_post(self):
        # 发送POST请求
        self.client.post("/request", json={"data": "test data"})