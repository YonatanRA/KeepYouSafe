import cv2
from imutils.video import VideoStream
from tensorflow.keras.models import load_model

from flask import Flask, render_template, Response, request

from detecta import Detector

import time



# iniciaciliza la aplicacion
app=Flask(__name__)

cam=VideoStream(0).start()

width=800   # pixeles imagen

modo='Camara'  # modo camara

modelo=cv2.dnn.readNetFromCaffe('detector/SSD_MobileNet_prototxt.txt', 'detector/SSD_MobileNet.caffemodel')
    
net_cara=cv2.dnn.readNet('detector/deploy.prototxt', 'detector/res10_300x300_ssd_iter_140000.caffemodel')

net_mascara=load_model('detector/detector_mascara.model')

detector=Detector(cam, width, modelo, net_cara, net_mascara)

    
    
@app.route('/tomar_foto', methods=['POST'])
def tomar_foto():
    foto=detector.foto
    cv2.imwrite('static/images/foto.png', foto)
    return Response(status=200)    
   
    

@app.route('/video')
def video():
    global modo

    if modo=='Mascara':
        return Response(detector.mascara(), mimetype='multipart/x-mixed-replace; boundary=frame')

    elif modo=='Distancia':
        return Response(detector.distancia(), mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        return Response(detector.camara(), mimetype='multipart/x-mixed-replace; boundary=frame')



    
@app.route('/', methods=['POST', 'GET'])
def main():
	time.sleep(1)
	if request.method=='POST':
		global modo
		modo=request.form['modo']
		return render_template('index.html', modo=modo)
	else:
		return render_template('index.html', modo=modo)
    
    
if __name__=='__main__':
    app.run(debug=False)
