import sys,os,json,io
import openai
import numpy as np
import onnxruntime
import base64
from PIL import Image


import boto3
s3_client = boto3.client("s3")
S3_BUCKET_NAME = 'leolambdalayer'
object_key = "model.onnx"  # replace object key
model = s3_client.get_object(
    Bucket=S3_BUCKET_NAME, Key=object_key)["Body"].read()



from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,ImageMessage
)


# 請將在組態中設定這些密鑰，並由環境變數取得
Channel_access_token = os.getenv('Channel_access_token',None)
Channel_Secret = os.getenv('Channel_Secret',None)
Openai_key = os.getenv('openai_key',None)


line_bot_api = LineBotApi(Channel_access_token)
handler = WebhookHandler(Channel_Secret)


def lambda_handler(event, context):

    # Set the OpenAI API endpoint URL and the secret key
    openai.api_endpoint = 'https://api.openai.com/v1/chat/completions'
    openai.api_key = Openai_key     
    
    label_dict = {'0': '擦傷', '1': '瘀傷', '2': '燒傷', '3': '刀傷', '4': '內生指甲', '5': '撕裂傷', '6': '刺傷'}
    
    @handler.add(MessageEvent, message=ImageMessage)
    def handle_image_message(event):
        SendImage = line_bot_api.get_message_content(event.message.id)

        image = Image.open(io.BytesIO(SendImage.content) )

        np_image = preprocess_image(image,224)
        
        ort_session = onnxruntime.InferenceSession(model)
        ort_inputs = {ort_session.get_inputs()[0].name: np_image}
        ort_outs = ort_session.run(None, ort_inputs)
        wound = np.where(np.max(ort_outs[0]))[0][0]

        msg=[
                {"role": "system", "content": "你是一位台灣的急救醫生,但目前無法前往現場,你將透過user的繁體字敘述指導對方遠程進行急救"},
                {"role": "user", "content": "目前患者嚴重出血，請問該如何處置?"},
                {"role": "assistant", "content": "1.立刻以直接加壓止血法止血。\
                2.若傷口有異物或斷肢，勿壓迫或去除之，應以環形墊圈固定包紮，送醫。"},
                {"role": "user", "content": "我現在有{},請問醫生我該如何處置?請用繁體字回答".format(label_dict[str(wound)])}]
        print(msg[-1])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msg
            )

        response = response["choices"][0]["message"]["content"]
          
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
            )

    @handler.add(MessageEvent, message=TextMessage)
    def handle_text_message(event):
        msg=[
                {"role": "system", "content": "你是一位台灣的急救醫生,但目前無法前往現場,你將透過user的繁體字敘述指導對方遠程進行急救"},
                {"role": "user", "content": "林小姐因感情糾紛服用大量漂白水自殺，目前應如何緊急處置?"},
                {"role": "assistant", "content": "1.應保持呼吸道通暢。2.令病人禁食。3.勿給任何中和劑。"},
                {"role": "user", "content": "目前患者嚴重出血，請問該如何處置?"},
                {"role": "assistant", "content": "1.立刻以直接加壓止血法止血。2.使患者靜臥，預防休克。\
                3.抬高出血部位，露出傷口，並覆蓋傷口，以防感染。4.勿去除血凝塊，持續出血時，繼續以消毒紗布加壓止血。\
                5.若有斷肢，須以無菌紗布包裏，置容器（或塑膠袋）中，外加冰塊及少許食鹽，以隨同患者送醫(最好在６－８小時內送醫)。\
                6.若傷口有異物或斷肢，勿壓迫或去除之，應以環形墊圈固定包紮，送醫。"},
                {"role": "user", "content": "{},請問醫生我該如何處置?請用繁體字回答".format(event.message.text)}]
        print(msg[-1])
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=msg
            )

        response = response["choices"][0]["message"]["content"]
        
        line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response))
    
    # get X-Line-Signature header value
    signature = event['headers']['x-line-signature']
    # get request body as text
    body = event['body']
    
    
    jug = json.loads(body)
    if jug['events'][0]['message']['type'] == "image":
        handler.handle(body, signature)
    else:
        handler.handle(body, signature)
        
    return {
        'statusCode': 200,
        'body': json.dumps("Hello from Lambda!")
        }
 
def center_crop(image, new_width, new_height):
    left = int(image.size[0]/2-new_width/2)
    upper = int(image.size[1]/2-new_height/2)
    right = left +new_width
    lower = upper + new_height
    return image.crop((left, upper,right,lower))


def preprocess_image(b64image, model_expected_im_size):
    image = b64image 
    #Image.open(io.BytesIO(b64image) )# 

    smallest_side = min(image.width, image.height)

    if smallest_side >= model_expected_im_size:
        maxwidth = model_expected_im_size
        maxheight = model_expected_im_size
        i = min(maxwidth/image.width, maxheight/image.height)
        a = max(maxwidth/image.width, maxheight/image.height)
        image.thumbnail((maxwidth*a/i, maxheight*a/i), Image.ANTIALIAS) # Antialias might be slow, can try removing
    else:
        # scale up
        scale_factor = (model_expected_im_size / smallest_side)
        image = image.resize((int(image.width*scale_factor), int(image.height*scale_factor)))

    # Center crop
    image = center_crop(image, model_expected_im_size, model_expected_im_size)
    image_np = np.array(image)

    if image.mode != "RGB":
        if(len(image_np.shape)<3):
            # Grayscale
            rgbimg = Image.new("RGBA", image.size)
            rgbimg.paste(image)
            image_np = np.array(rgbimg)
            image_np = image_np[...,:3]
        else:
            # Other (RGBA)
            image_np = image_np[...,:3]

    # Use to debug crop/scale issues
    #Image.fromarray(image_np.astype(np.uint8)).save("cropped_test_image.png")

    # Normalize for input to efficientNet
    # image_np = image_np / 255
    # image_np = image_np - [0.485, 0.456, 0.406]
    # image_np = image_np / [0.229, 0.224, 0.225]

    # Channels goes first, not last
    image_np = np.moveaxis(image_np, -1, 0)

    # Add batch dimension to the front
    image_np = image_np[np.newaxis, ...]
    return image_np.astype(np.float32)
