# Example using luma.oled (SSD1306) on Raspberry Pi 5B via I²C

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont
import time

# Initialize I²C (bus=1, address=0x3C is common for 128×64 OLED)
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Create a blank image buffer
width, height = device.width, device.height
image = Image.new("1", (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Draw text (or shapes) onto the buffer
draw.rectangle((0, 0, width - 1, height - 1), outline=0, fill=0)  # clear display
draw.text((0, 0), "Hello, Pi 5B!", font=font, fill=255)

# Send buffer to OLED
device.display(image)
time.sleep(5)
