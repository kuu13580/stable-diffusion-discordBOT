import discord
from configparser import ConfigParser

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

ini = ConfigParser()
ini.read("./settings.ini", encoding="utf-8")
config = ini["DEFAULT"]

@client.event
async def on_ready():
    channel = client.get_channel(int(config["on_ready_channel_id"]))  # メッセージを送信するチャンネルのIDを設定
    await channel.send('Bot is disconnected.')  # メッセージを送信
    await client.close()  # ログアウト

client.run(config["discord_token"])  # ログインに必要なトークンを設定