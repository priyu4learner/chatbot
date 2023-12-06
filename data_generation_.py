import bs4 as bs
import yaml
import urllib.request

headers = { 'accept':'*/*',
'accept-encoding':'gzip, deflate, br',
'accept-language':'en-GB,en;q=0.9,en-US;q=0.8,hi;q=0.7,la;q=0.6',
'cache-control':'no-cache',
'dnt':'1',
'pragma':'no-cache',
'referer':'https',
'sec-fetch-mode':'no-cors',
'sec-fetch-site':'cross-site',
'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
 }

def save_to_yaml_qa(qa_list, output_file='output_qa.yml'):
    with open(output_file, 'w') as yaml_file:
        yaml.dump(qa_list, yaml_file, default_flow_style=False)
def tpo_faq():
    get_link=urllib.request.urlopen('https://tpo.mnnit.ac.in/tnp/company/faqs.php')
    get_link=get_link.read()
    dat=bs.BeautifulSoup(get_link,'lxml')
    data=dat.find('div',id="accordion").find_all(class_='panel panel-default')
    #datas+=dat.find('div',id="accordion").text
    sen=[]
    for d in data:
        question = d.find('h4').text
        answer=d.find('div',class_='panel-body').text
        question=question.strip()
        answer=answer.strip()
        sen.append([question,answer])

    save_to_yaml_qa(sen,output_file='C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/my_data/placement.yml')

def faculty_info():
    get_link=urllib.request.urlopen('http://www.mnnit.ac.in/index.php/department/engineering/csed/csedfp')
    get_link=get_link.read()
    dat=bs.BeautifulSoup(get_link,'lxml')
    data=dat.find('table').find_all('tr')
    #datas=dat.find('table').text
    c=0
    sen=[]
    for d in data:
        if c<74 and (c%8==0 or c%8==1 ):
            v=d.find('td').text
            v=v.strip()
            sen.append(v)
        elif c>74 and (c%8==1 or c%8==2) :
            v=d.find('td').text
            v=v.strip()
            sen.append(v)
        c=c+1
    save_to_yaml_qa(sen,output_file='C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/my_data/faculty.yml')
faculty_info()  