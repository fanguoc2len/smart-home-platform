from pyngrok import ngrok

print(">>> Đang mở tunnel ngrok ...")
tunnel = ngrok.connect(5000, bind_tls=True)
print("✅ Public URL:", tunnel.public_url)