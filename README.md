StolenCarAlert is an alert system, base on YoloV2 (you can find the readme about it in the files).<br>

The YoloV2 AI was training on 1000 different images, specific to the Israeli license plates style, and reach a precision of almost 90%.

How it work?<br>
The system scan with camera and search for license plates,
in the very first frame with any vehicle, the license plate will be recognized,
cropped from the image and let the knn algorithem (also train specific on Israeli license plate's numbers) do his job and convert the cropped image to numbers.
From that moment, the system will send the number direct to the Isreali police offical website via https request, and the result will print on the screen.
<br><br>
<img src="https://github.com/Yogranov/StolenCarAlert/blob/master/README_MEDIA/demo.jpg" width="700" height="350" />
