# yolo-with-opencv
Bu reponun amacı openCV kütüphanesiyle yolo modellerini kullanmak.

Kodun içerisindeki kütüphaneler:
opencv, matplotlib, numpy.

Modelinizi kullanabilmek için modelin ağırlık(.weights), config(.cfg) ve isim(.names) dosyasına ihtiyacınız var. 

Main dosyası içerisindeki readNetFromDarknet() fonksiyonu içerisine kendi .weights dosyanızı ve .cfg dosyanızın ismini yazmanız gerekiyor.

classFile = 'obj.names' satırındaki .names dosyasının ismini de kendi .names dosyanızın ismiyle değiştirmeniz gerekiyor.

Son olarak cv2.imread('plaka.png') fonkiyonu içerisine de kendi fotoğrafınızı koyarak kodu çalıştırabilirsiniz.


Plaka modeli ağırlıkları:
https://drive.google.com/file/d/1tpI8mEozmE4PLwiuFAjkdnEXPuy86yUC/view?usp=sharing
