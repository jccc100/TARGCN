import smtplib
import email.mime.text
import email.mime.multipart
import datetime


import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


def get_type_file(path="./test/aa.npy", keyword='.npy'):  # 这里可以更改扩展名如.doc,.py,.zip等等
    # 打印当前的工作目录
    print("当前目录为: ", os.getcwd() + path)

    # 列举当前工作目录下的文件名

    files = os.listdir(path)

    keyword = keyword
    filelist = []

    i = 0
    for file in files:
        # print("aaa")
        if keyword in file:
            i = i + 1
            print(i, file)
            filelist.append(file)

    return filelist


def send_email(path, keyword='.pth', content=""):
    smtpHost = 'smtp.qq.com'
    sendAddr = 'XXX@qq.com'
    password = 'bxbzlopushpbfjjh'
    receiver = 'XXX@qq.com'
    subject = "训练文件"

    current_time=datetime.datetime.now()
    content = str(current_time)
    # try:
    #     from model.Run import args
    #     content +='\n'+args.dateset
    # except:
    #     pass

    msg = MIMEMultipart()
    msg['from'] = sendAddr
    msg['to'] = receiver
    msg['Subject'] = subject

    txt = MIMEText(content, 'plain', 'utf-8')
    msg.attach(txt)
    file_num = 1
    if os.path.isdir(path):
        files = os.listdir(path)
        filelist = get_type_file(path, keyword)
        file_num = str(len(filelist))
        filename = ""

        i = 0
        for file in filelist:
            i = i + 1
            filename = file
            # print(str(i),filename)
            part = MIMEApplication(open(path + filename, 'rb').read())
            part.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(part)
    else:
        part = MIMEApplication(open(path, 'rb').read())
        part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(path))
        msg.attach(part)
    # 添加附件,传送filelist列表里的文件

    server = smtplib.SMTP_SSL(smtpHost, 465)  # SMTP协议默认端口为25
    # server.set_debuglevel(1)  # 出错时可以查看

    server.login(sendAddr, password)
    server.sendmail(sendAddr, receiver, str(msg))
    print("\n", file_num, "个文件发送成功")
    server.quit()

if __name__=="__main__":
    path="send_email.py"
    send_email(path)