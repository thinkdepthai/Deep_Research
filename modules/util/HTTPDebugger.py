import requests
import logging
import http.client as http_client

logging.basicConfig()

requests_log = logging.getLogger("requests.packages.urllib3")



class HTTPDebugger():
    @staticmethod
    def enable():
        http_client.HTTPConnection.debuglevel = 1
        requests_log.propagate = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log.setLevel(logging.DEBUG)

    @staticmethod
    def disable():
        http_client.HTTPConnection.debuglevel = 0
        requests_log.propagate = False
        logging.getLogger().setLevel(logging.INFO)
        requests_log.setLevel(logging.INFO)
        

# run
if __name__ == '__main__':
    HTTPDebugger.enable()
    requests.get('https://httpbin.org/headers', timeout=10)
