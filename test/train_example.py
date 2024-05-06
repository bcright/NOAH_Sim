import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request():
    url = 'http://localhost:5000/request'  # 确保这是正确的URL
    data = {'data': 'sample'}  # 根据你的API调整
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        return str(e)

def main():
    # num_requests = 100  # 你想发送的请求总数
    with ThreadPoolExecutor(max_workers=10) as executor:
        while True:  # 使用无限循环来持续发送请求
            futures = [executor.submit(send_request) for _ in range(10)]  # 每次循环发送10个请求
            for future in as_completed(futures):
                print(future.result())

if __name__ == '__main__':
    main()