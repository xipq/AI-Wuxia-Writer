from utils import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from train import T5_Model


def test(book):
    model_path = './ckpt'

    model = T5_Model('/home/cjx/lyx/T5-NLP/t5_nlp/nlg_task/ckpt/model_big25')
    tokenizer = T5PegasusTokenizer.from_pretrained('/home/cjx/lyx/T5-NLP/t5_nlp/nlg_task/ckpt/tokenizer_big25')

    i = 1
    for text in book:
        ids = tokenizer.encode(text, return_tensors='pt')
        #print("Raw:", ids)
        output = model.generate_test(ids)
        predict = tokenizer.batch_decode(output, skip_special_tokens=True)
        print("Text: ", i)
        i += 1
        print("Input: ", text)
        print("Output: ", ''.join(predict).replace(' ', ''))


def save_test(book, model_name):
    model_dir = '/home/cjx/lyx/T5-NLP/t5_nlp/nlg_task/ckpt/model_' + model_name
    tokenizer_dir = '/home/cjx/lyx/T5-NLP/t5_nlp/nlg_task/ckpt/tokenizer_' + model_name

    file = open("./result/" + model_name + ".txt", "w", encoding="utf-8")

    model = T5_Model(model_dir)
    tokenizer = T5PegasusTokenizer.from_pretrained(tokenizer_dir)

    i = 1
    for text in book:
        ids = tokenizer.encode(text, return_tensors='pt')
        #print("Raw:", ids)
        output = model.generate_test(ids)
        predict = tokenizer.batch_decode(output, skip_special_tokens=True)
        print("Text: ", i)
        i += 1
        print("Input: ", text)
        str_output = ''.join(predict).replace(' ', '')
        print("Output: ", str_output)
        file.write(str_output)
        file.write("\n")
    
    print("Results documented.")
    file.close()



if __name__ == "__main__":
    book = ["忽见山坡下寺院边门中冲出七八名僧人，手提齐眉木棍，吆喝道：",
        "她一时沉吟未决，蓦地里眼前黄影晃动，一人喝道：“到少林寺来既带剑，又伤人，世上焉有是理？”",
        "秦淮河花舫笙歌，聚六朝金粉，此时已是子夜，但寻欢逐乐的公子阔少仍未散尽，",
        "船里的人都跑了出来，那小女孩尖声叫着姐姐，不一会从后舱走出一个年纪亦不太大的少女，",
        "那小女孩看了，不禁拉了拉她姐姐的衣角，低声说着：",
        "如今江湖已乱，死的人已有不少，七大子弟已死伤殆尽——白衣人是否重来，犹末可知，",
        "宝儿瞧得清楚，此人竞赫然正是那火魔鬼——他那双火也似的妖异目光，宝儿永生再也不会忘记。",
        "只听四面一阵阵欢呼，只要欧阳天矫一招攻击；四面便必定有人为他喝采、助威，想来他门下弟子前来观战的，必有不少。",
        "哪知她身子虽然斜斜向前倒下，双足却紧紧钉在擂台上，整个人就像是根标枪似的，斜插在擂台边缘。",
        "火魔神与他门下本已施展身形，要待冲上台去，听得这声惊呼，脚步不由得为之一顿",
        "芮玮停剑喘了一口气，只见他脸色煞白，一颗心自抨抨直跳不止，心想，好险！好险！要是再被老道攻破最后一道剑幕，非被老道刺伤不可！",
        "芮玮功力虽大不如老道，凭丈海渊剑法的精妙，把老道击出来的剑势一一化解，只见两方一时战个平手。",
        "转眼过了数日，已是中秋。这日午后，胡斐带同程灵素、蔡威、姬晓峰三人，径去福康安府中，赴那天下武林掌门人大会。",
        "程灵素见他若有所思，目光中露出温柔的神色，早猜到他是在想起了袁紫衣，心中微微一酸，忽见他颊边肌肉一动，脸色大变，双眼中充满了怒火，",
        "程灵素道：“她还没来。”胡斐明知她说的是袁紫衣，却顺口道：",
        "这汤沛一走进大厅，真便似“大将军八面威风”，人人的眼光都望着他。",
        "我从小就不喜欢练剑。",
        "果然他这句话刚说完，人丛中同时走出两个人来，在两张椅中一坐。一个大汉身如铁塔，一言不发，却把一张紫檀木的太师椅坐得格格直响。",
        "一个身材高瘦的汉子踉踉跄跄而出，一手拿酒壶，一手拿酒杯，走到厅心，晕头转向的绕了两个圈子，突然倒转身子，向后一跌，摔入了那只空椅之中。",
        "黄希节的儿子一刀向对手剁去，却剁了个空。海兰弼一伸手，抓住他的胸口，顺手向外掷出，跟着回手抓住宗雄的弟子，也掷到了天井之中。",
        "两人并肩站在黑暗之中，默然良久，忽听得屋瓦上喀的一声响。胡斐大喜，只道袁紫衣去而复回，情不自禁的叫道：“你……你回来了！”",
        "忽听得背后脚步之声细碎，隐隐香风扑鼻，他回过身来，见是一个美貌少妇，身穿淡绿纱衫，含笑而立，正是马春花。",
        "突觉背后金刃掠风，一人娇声喝道：“手下留人！”喝声未歇，刀锋已及后颈。这一下来得好快，",
        "袁紫衣格格娇笑，倒转匕首，向他掷了过去，跟着自腰间撤出软鞭，笑道：“胡大哥，咱们真刀真枪的较量一场。” 胡斐正要伸手去接匕首，",
        "胡斐知道程灵素决不是她敌手，此刻若去追杀凤天南，生怕袁紫衣竟下杀手，纵然失去机缘，也只得罢了，当下跃进园中，挺刀叫道：",
        "袁紫衣一双妙目望定胡斐，说道：“你怎么不刺？”忽听得曾铁鸥叫道：",
        "袁紫衣笑道：“你不说我也知道，你的功夫不如他们，我要挑弱的先打，好留下力气，对付强的。外边草地上滑脚，咱们到亭中过招。上来吧！”",
        "身形一晃，进了亭子，双足并立，沉肩塌胯，五指并拢，手心向上，在小腹前虚虚托住，正是“八极拳”的起手式“怀中抱月”。",
        "秦耐之脸上一红，更不答话，弯腰跃进亭中，一招“推山式”，左掌推了出去。袁紫衣摇了摇头，说道：“这招不好！”更不招架，只是向左踏了一步，"]
    #test(book)
    models = ['big25', 'big220']

    for model in models:
        save_test(book, model)