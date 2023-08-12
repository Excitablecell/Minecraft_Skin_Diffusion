# 亢体造梦 2023 copyright
from diffusers import DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt
import minepi
import os
import asyncio

def count_files(directory):
    '''
    计算目录下文件数量
    :param directory: 目录
    :return: 文件数量
    '''
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

async def render_minecraft_skin(minecraft_skin):
    await minecraft_skin.render_skin()
    return minecraft_skin.skin


if __name__ == '__main__':
    # 使用huggingface的diffusers库渲染皮肤
    pipeline = DDPMPipeline.from_pretrained('./mcskin_diffuser_0609').to('cuda')
    image = pipeline().images[0].convert('RGBA')

    # image = Image.open("./results/skin16.png")
    # pixdata = image.load()

    pixdata = image.load()
    # 逐个像素判断是否为黑色，若为黑色则转换成透明像素
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if pixdata[x, y][3] < 115:
                pixdata[x, y] = (255, 255, 255, 0)
            image.putpixel((x, y), pixdata[x, y])

    # 保存皮肤
    files_num = count_files(r'./') + 1
    image.save('./skin' + str(files_num) + r'.png')

    # 使用minepi库渲染并生成皮肤
    skin_object = minepi.Skin(raw_skin=image, raw_cape=None, raw_skin_url=None, raw_cape_url=None, name=None)
    asyncio.run(render_minecraft_skin(skin_object))
    skin_object.skin.show()