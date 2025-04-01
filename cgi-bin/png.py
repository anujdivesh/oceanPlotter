#!/usr/bin/python3
import matplotlib.pyplot as plt 
import sys
import os
from io import StringIO


# Creating list of Month and Share_buy for Plotting Line Graph 
Month = ['January', 'February', 'March'] 
Share_buy = [10, 17, 30] 
  
# Plotting Line Graph 
plt.title("Share's Buy in a month") 
plt.xlabel('Months') 
plt.ylabel("No of Share's") 
plt.plot(Month, Share_buy) 

format = "png"
sio = StringIO()
plt.savefig(sio, format=format)

data_uri = sio.getvalue().encode('base64').replace('\n', '')
img_tag = '<img src="data:image/png;base64,{0}" alt="sucka" />'.format(data_uri)

print("Content-type: text/html\n")
print("<title>Try Ageen</title>")
print("<h1>Hi</h1>")
print(img_tag)

# Saving a plotted graph as an Image 
#plt.savefig('plot_image.png') 
#plt.show() 
#print(cwd)
#sys.stdout.write( "Content-type: image/png\r\n\r\n" + file(cwd+'/plot_image.png',"rb").read() )
#print("Content-Type: text/html; charset=UTF-8")
#print("")
#print("Hello to FastCGI - Anuj"+cwd)
