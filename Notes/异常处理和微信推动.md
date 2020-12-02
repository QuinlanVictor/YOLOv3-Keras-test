"""
1202   测试一下微信推送和异常处理机制


"""

import requests
import traceback
sckey = 'SCU131802Tef40bc6617c6e29c898cfdc99dbcbcc55fc655fa91537'#在发送消息页面可以找到

# #url = 'https://sc.ftqq.com/%s.send?text=程序完成了'%sckey
# #text为推送的title,desp为推送的描述
# url = 'https://sc.ftqq.com/%s.send?text=训练结束！&desp=train1202'%sckey
# requests.get(url)

def _main():

    a = 10 * (1/0)
    print(a)



if __name__ == '__main__':
    try:
        _main()
        url = 'https://sc.ftqq.com/%s.send?text=训练成功！&desp=train1202' % sckey
        requests.get(url)
    except Exception as e:
        print('异常类型：',str(e))
        traceback.print_exc()
        # url = 'https://sc.ftqq.com/%s.send?text=train1202&desp=训练失败！' % sckey
        # requests.get(url)
