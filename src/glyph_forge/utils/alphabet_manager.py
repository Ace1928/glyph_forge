import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union


class AlphabetCategory(Enum):
    """Categorization of different alphabet types"""

    DENSITY = "density"  # For grayscale density mapping
    SPECIAL = "special"  # For special character collections
    SYMBOLIC = "symbolic"  # For symbolic representation
    ARTISTIC = "artistic"  # For artistic effects
    LANGUAGES = "languages"  # For language-specific characters


# Collection of character sets for different conversion purposes
ALPHABETS = {
    # Density-based alphabets (dark to light)
    "general": " .,:;i1tfLCG08@",  # General purpose, 14 levels
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|/''tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",  # Detailed, 70 levels
    "simple": "@%#*+=-:. ",  # Simple, 10 levels (reversed)
    "blocks": " ░▒▓█",  # Unicode block characters, 5 levels
    "binary": "01",  # Binary, 2 levels
    "dots": " ⠄⠆⠖⠶⡶⣩⣪⣫",  # Braille patterns, 10 levels
    "eidosian": "⚡✨🔥💫🌟⭐⚪⭕⚫",  # Special Eidosian alphabet, 9 levels
    "cosmic": "✧·˚✫⋆˚。⋆｡⋆☾˚⋆✦✩★",  # Cosmic symbols, 12 levels
    "matrix": "日+*%$#@&01",  # Matrix-like characters, 10 levels
    "shadows": " ░▒▓▀▁▂▃▄▅▆▇█",  # Shadows with gradients, 13 levels
    "minimal": " .-=+*#%@",  # Minimal set, 9 levels
    "contrast": " █",  # Maximum contrast, 2 levels
    "extended": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$▀▄█",  # Extended with blocks, 73 levels
    "ethereal": "·°•○◎●◉✷✸✹✺✻✼❂⊛⊕⦿",  # Ethereal patterns, 16 levels
    "quantum": "⟨⟩⟪⟫⦑⦒⦓⦔⦦⦧⦨⦩⦪⦫⦬⦭⦮⦯⦰⦱⦲⦳⦴⦵⦶⦷⦸⦹⦺⦻⦼⦽⦾⦿",  # Quantum-like symbols, 17 levels
    "dimensional": "⎲⎳⏏⏑⏒⏣⏤⏥⎔⎕⌘⌑⌓⌭⌬",  # Dimensional symbols, 15 levels
}

# Specialized character sets grouped by purpose
SPECIAL_SETS = {
    # Structural elements
    "box_drawing": "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬╒╓╕╖╘╙╛╜╞╟╡╢╤╥╧╨╪╫",
    "blocks": "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓",
    "braille": "⠀⠁⠂⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾⠿",
    "arrows": "←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻",
    "math": "∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿",
    # Eidosian specialized sets
    "eidosian_energy": "⚡⚪⭕⚫✨⭐★☀︎⚝✴️✫☉☼☄︎",
    "eidosian_cosmic": "☽☾★✩✧⋆✫✬✭✮✯✰⚝✢✣✤✥❂❈❉❊❋✱✲✳✴✵✶✷✸✹✺✻✼❄❅❆❇❈❉❊❋",
    "eidosian_mystic": "⚕♱♰☤⚚⚖⚘⚛⚜⚝⚢⚣⚤⚥⚦⚧⚨",
    "eidosian_elements": "🔥💧🌪️⛰️✨💫⭐",
    # Additional specialized sets
    "geometric": "■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯▰▱▲△▴▵▶▷▸▹►▻▼▽▾▿◀◁◂◃◄◅◆◇◈◉◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯",
    "shapes": "■□▢▣▤▥▦▧▨▩◆◇◈◊○◌◍◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯",
    "weather": "☀☁☂☃☄★☆☇☈☉☊☋☌☍☐☑☒☓☔☕☖☗☘☙☚☛☜☞☟☠☡☢☣☤☥☦☧☨☩☪☫☬☭☮☯☰☱☲☳☴☵☶☷☸☹☺☻☼☽☾☿♀♁♂♃♄♅♆♇",
    "zodiac": "♈♉♊♋♌♍♎♏♐♑♒♓",
    "tarot": "⚀⚁⚂⚃⚄⚅♠♡♢♣♤♥♦♧♨♩♪♫♬♭♮♯",
    "chess": "♔♕♖♗♘♙♚♛♜♝♞♟",
    "planets": "☿♀♁♂♃♄♅♆♇",
    "circuit": "⏚⏛⎓⌁⌂⌀⏃⏄⏅⏆⏇⏈⏉⏊⎌⎍⎎⎏⎐⎑⎒",
    "musical": "♩♪♫♬♭♮♯𝄞𝄟𝄠𝄡𝄢𝄣𝄤𝄥𝄦𝄫𝄬𝄪",
    "domino": "🀱🀲🀳🀴🀵🀶🀷🀸🀹🀺🀻🀼🀽🀾🀿🁀🁁🁂🁃🁄🁅🁆🁇🁈🁉🁊🁋🁌🁍🁎🁏🁐🁑🁒🁓🁔🁕🁖🁗🁘🁙🁚🁛🁜🁝🁞🁟🁠🁡",
    "runes": "ᚠᚡᚢᚣᚤᚥᚦᚧᚨᚩᚪᚫᚬᚭᚮᚯᚰᚱᚲᚳᚴᚵᚶᚷᚸᚹᚺᚻᚼᚽᚾᚿᛀᛁᛂᛃᛄᛅᛆᛇᛈᛉᛊᛋᛌᛍᛎᛏᛐᛑᛒᛓᛔᛕᛖᛗᛘᛙᛚᛛᛜᛝᛞᛟᛠᛡᛢᛣᛤᛥᛦᛧᛨᛩᛪ",
}

LANGUAGES = {
    "english": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?\"'`()[]{}@#$%^&*_+-=<>/\\|~",
    "cyrillic": "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,;:!?\"'`()[]{}№",
    "greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ0123456789.,;:!?\"'`()[]{}·",
    "arabic": "ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآءؤةى٠١٢٣٤٥٦٧٨٩،؛؟.!\"'`()[]{}",
    "hebrew": "אבגדהוזחטיכךלמםנןסעפףצץקרשת0123456789.,;:!?\"'`()[]{}״׳",
    "korean": "가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궃궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난닫날낡낢남납낫낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넝넣네넥넨넬넴넵넷넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐림립닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둤둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫃뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섁섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝",
    "japanese": "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをんゃゅょっーアイウエオカキクケコガギグゲゴサシスセソザジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヲンャュョッー、。・「」『』！？…あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
    "chinese": "一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億钱人大中小天地日月北南东西山川水火木土金石风云雨雪龙凤鸟虫鱼草花米面肉心手口目耳鼻头发脸皮衣裤袜鞋帽车船飞机票屋室门窗床桌椅板凳书报纸笔墨毛巾肥皂牙刷杯茶盘碗筷勺子湖海江河秋冬上下多少远近左右前后里外男女老少声音出入看听说读写吃喝玩乐爱恨怒哀愁忧烦闷思念盼等待勤劳苦累休闲干净脏乱静闹胖瘦高矮美丑好坏冷暖忙闲新旧深浅长短足快慢坐立行住特产活熱語字問說話事情",
    "thai": "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์๎๏๐๑๒๓๔๕๖๗๘๙๚๛.,;:!?\"'`()[]{}",
    "vietnamese": "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọơớờởỡợôốồổỗộpqrstuúùủũụưứừửữựvwxyýỳỷỹỵz0123456789.,;:!?\"'`()[]{}",
    "armenian": "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև0123456789.,;:!?\"'`()[]{}",
    "georgian": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ0123456789.,;:!?\"'`()[]{}",
    "bengali": "অআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৸৹৺.,;:!?\"'`()[]{}",
    "tamil": "அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநனபமயரறலளழவஶஷஸஹஜஷஸஹாிீுூெேைொோௌ்ௐௗ௧௨௩௪௫௬௭௮௯௰௱௲.,;:!?\"'`()[]{}",
    "telugu": "అఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౘౙౠౡౢౣ౦౧౨౩౪౫౬౭౮౯.,;:!?\"'`()[]{}",
    "kannada": "ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೠೡೢೣ೦೧೨೩೪೫೬೭೮೯.,;:!?\"'`()[]{}",
    "malayalam": "അആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃൄെേൈൊോൌ്ൗൠൡൢൣ൦൧൨൩൪൫൬൭൮൯.,;:!?\"'`()[]{}",
    "punjabi": "ਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸ਼ਸਹਾਿੀੁੂੇੈੋੌ੍ੰੱੲੳ੦੧੨੩੪੫੬੭੮੯.,;:!?\"'`()[]{}",
    "burmese": "ကခဂဃငစဆဇဈဉညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအာိီုူေဲါာံ့း္၀၁၂၃၄၅၆၇၈၉၊။၌၍၎၏.,;:!?\"'`()[]{}",
    "khmer": "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអាិីឹឺុូួើឿៀេែៃោៅំះុំ៎ាំ៝០១២៣៤៥៦៧៨៩.,;:!?\"'`()[]{}",
    "lao": "ກຂຄງຈຊຍດຕຖທນບປຜຝພຟມຢຣລວສຫອຮວະັາິີຶືຸູົຼຽເແໂໃໄ່້໊໋໌ໍ໐໑໒໓໔໕໖໗໘໙.,;:!?\"'`()[]{}",
    "sinhala": "අආඇඈඉඊඋඌඍඎඏඐඑඒඓඔඕඖකඛගඝඞඟචඡජඣඥඦටඨඩඪණඬතථදධනඳපඵබභමඹයරලවශෂසහළෆාැෑිීුූෘෙේෛොෝෞං.,;:!?\"'`()[]{}",
    "sundanese": "ᮃᮄᮅᮆᮇᮈᮉᮊᮋᮌᮍᮎᮏᮐᮑᮒᮓᮔᮕᮖᮗᮘᮙᮚᮛᮜᮝᮞᮟᮠᮡᮢᮣᮤᮥᮦᮧᮨᮩ᮪᮫ᮬᮭ᮰᮱᮲᮳᮴᮵᮶᮷᮸᮹.,;:!?\"'`()[]{}",
    "javanese": "ꦄꦅꦆꦇꦈꦉꦊꦋꦌꦍꦎꦏꦐꦑꦒꦓꦔꦕꦖꦗꦘꦙꦚꦛꦜꦝꦞꦟꦠꦡꦢꦣꦤꦥꦦꦧꦨꦩꦪꦫꦬꦭꦮꦯꦰꦱꦲ꦳ꦴꦵꦶꦷꦸꦹꦺꦻꦼꦽꦾꦿ꧀꧁꧂꧃꧄꧅꧆꧇꧈꧉꧊꧋꧌꧍꧎ꧏ꧐꧑꧒꧓꧔꧕꧖꧗꧘꧙.,;:!?\"'`()[]{}",
}


ALPHABET_CATEGORIES: Dict[Union[str, AlphabetCategory], List[str]] = {}


def populate_alphabet_categories(
    *category_sources: Mapping[Union[str, AlphabetCategory], Any],
) -> None:
    """
    Procedurally build ALPHABET_CATEGORIES from any number of sources.
    Each source can map a category (any key) to various data structures (lists, dicts, etc.).
    This function merges them into ALPHABET_CATEGORIES, avoiding duplicates.
    """
    for source in category_sources:
        for cat_key, data in source.items():
            if cat_key not in ALPHABET_CATEGORIES:
                ALPHABET_CATEGORIES[cat_key] = []

            # Handle various data structures that might be in sources
            if isinstance(data, dict):
                ALPHABET_CATEGORIES[cat_key].extend(list(data.keys()))
            elif isinstance(data, list):
                ALPHABET_CATEGORIES[cat_key].extend(data)
            else:
                # Single item
                ALPHABET_CATEGORIES[cat_key].append(str(data))

            # Deduplicate entries
            ALPHABET_CATEGORIES[cat_key] = list(set(ALPHABET_CATEGORIES[cat_key]))


# Initialize categorization - call this at module load time
populate_alphabet_categories(
    {AlphabetCategory.DENSITY: ALPHABETS},
    {AlphabetCategory.SPECIAL: SPECIAL_SETS},
    {AlphabetCategory.LANGUAGES: LANGUAGES},
)


class AlphabetManager:
    """Manager for alphabets, special sets, and languages."""

    @staticmethod
    def get_alphabet(name: str = "general") -> str:
        """
        Retrieve a named alphabet, special set, or language;
        default to 'general' if not found.
        """
        if name in ALPHABETS:
            return ALPHABETS[name]
        elif name in SPECIAL_SETS:
            return SPECIAL_SETS[name]
        elif name in LANGUAGES:
            return LANGUAGES[name]
        return ALPHABETS["general"]

    @staticmethod
    def get_special_set(name: str) -> Optional[str]:
        return SPECIAL_SETS.get(name)

    @staticmethod
    def register_alphabet(
        name: str, charset: str, category: AlphabetCategory = AlphabetCategory.DENSITY
    ) -> None:
        ALPHABETS[name] = charset
        if category not in ALPHABET_CATEGORIES:
            ALPHABET_CATEGORIES[category] = []
        if name not in ALPHABET_CATEGORIES[category]:
            ALPHABET_CATEGORIES[category].append(name)

    @staticmethod
    def register_special_set(
        name: str, charset: str, category: AlphabetCategory = AlphabetCategory.SPECIAL
    ) -> None:
        SPECIAL_SETS[name] = charset
        if category not in ALPHABET_CATEGORIES:
            ALPHABET_CATEGORIES[category] = []
        if name not in ALPHABET_CATEGORIES[category]:
            ALPHABET_CATEGORIES[category].append(name)

    @staticmethod
    def create_density_map(charset: str) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        length = len(charset)
        for i in range(256):
            idx = min(int(i * length / 256), length - 1)
            mapping[i] = charset[idx]
        return mapping

    @staticmethod
    def create_custom_density_map(
        charset: str, min_value: int = 0, max_value: int = 255, reverse: bool = False
    ) -> Dict[int, str]:
        if reverse:
            charset = charset[::-1]
        length = len(charset)
        mapping: Dict[int, str] = {}
        size = max_value - min_value + 1
        for i in range(min_value, max_value + 1):
            rel = i - min_value
            idx = min(int(rel * length / size), length - 1)
            mapping[i] = charset[idx]
        return mapping

    @staticmethod
    def list_available_alphabets() -> List[str]:
        return list(ALPHABETS.keys())

    @staticmethod
    def list_special_sets() -> List[str]:
        return list(SPECIAL_SETS.keys())

    @staticmethod
    def list_by_category(category: AlphabetCategory) -> List[str]:
        return ALPHABET_CATEGORIES.get(category, [])

    @staticmethod
    def get_category(name: str) -> Optional[AlphabetCategory]:
        for cat, names in ALPHABET_CATEGORIES.items():
            if name in names and isinstance(cat, AlphabetCategory):
                return cat
        return None

    @staticmethod
    def combine_alphabets(names: List[str]) -> str:
        result = ""
        seen: set[str] = set()
        for n in names:
            for c in AlphabetManager.get_alphabet(n):
                if c not in seen:
                    seen.add(c)
                    result += c
        return result

    @staticmethod
    def filter_charset(charset: str, pattern: str) -> str:
        return "".join(c for c in charset if re.match(pattern, c))

    @staticmethod
    def get_charset_info(
        name: str,
    ) -> Dict[str, Union[str, int, AlphabetCategory, bool, None]]:
        cs = AlphabetManager.get_alphabet(name)
        cat = AlphabetManager.get_category(name)
        return {
            "name": name,
            "charset": cs,
            "length": len(cs),
            "category": cat,
            "is_special": name in SPECIAL_SETS,
        }

    @staticmethod
    def get_weighted_charset(base_charset: str, weights: List[float]) -> str:
        if len(base_charset) != len(weights):
            raise ValueError("Character set and weights must have the same length.")
        result = ""
        for ch, w in zip(base_charset, weights, strict=True):
            result += ch * max(1, int(w * 10))
        return result

    @staticmethod
    def invert_charset(charset: str) -> str:
        return charset[::-1]

    @staticmethod
    def create_interpolated_charset(
        charset1: str, charset2: str, ratio: float = 0.5
    ) -> str:
        ratio = max(0.0, min(1.0, ratio))
        len1, len2 = len(charset1), len(charset2)
        target_len = max(3, int(len1 * (1 - ratio) + len2 * ratio))
        result = ""
        for i in range(target_len):
            pos = i / (target_len - 1)
            if pos < ratio and ratio > 0:
                idx = int(pos / ratio * (len2 - 1))
                result += charset2[idx]
            else:
                rel = ((pos - ratio) / (1 - ratio)) if ratio < 1 else 0
                idx = int(rel * (len1 - 1))
                result += charset1[idx]
        return result
