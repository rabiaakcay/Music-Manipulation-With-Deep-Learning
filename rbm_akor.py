# 2020 Bahar Bitirme Çalışması 330099 - 330123
import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

###################################################
# Kodun çalışması için, bu dosyayı midi_manipulation.py dosyası ve
# Pop_Music_Midi klasörü ile aynı yola koyulmalıdır.

import midi_manipulation
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sarki_al(path):
    dosyalar = glob.glob('{}/*.mid*'.format(path))
    sarkilar = []
    for f in tqdm(dosyalar):
        try:
            sarki = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(sarki).shape[0] > 50:
                sarkilar.append(sarki)
        except Exception as e:
            raise e
    return sarkilar


sarkilar = sarki_al('Pop_Music_Midi')  # Şarkılar midi'den msgpack'e çevrilmiştir
print("{} sarkilar islendi".format(len(sarkilar)))
###################################################

# Hiperparametreler
# Öncelikle modelimiz için hiperparametrelere bakalım:

en_dusuk_nota = midi_manipulation.lowerBound  # Piyano düzenindeki en düşük notanın indexi
en_yuksek_nota = midi_manipulation.upperBound # Piyano düzenindeki en düşük notanın indexi
nota_araligi = en_yuksek_nota - en_dusuk_nota  # nota aralığı

zaman_adimlari_sayisi = 15  # bir zamanda oluşturacağımız zaman adımlarının sayısı
gorulebilen_katman_boyutu = 2 * nota_araligi * zaman_adimlari_sayisi  # görünebilen katmanın boyutu
gizli_katman_boyutu = 50  # gizli katmanın boyutu

epochlarin_sayisi = 200  # Koşacağımız eğitim epochlarinin sayısı
# her epoch için tüm datasete bakacak
batch_boyutu = 100  # RBM aracılığıyla belirli bir zamanda göndereceğimiz eğitim örnekleri
derece = tf.constant(0.005, tf.float32)  # modelimizin öğrenme derecesi

# Değişkenler:
# Sırada kullanacaığımız değişkenlere bakabiliriz:
# verimizi tutan placeholder değişkeni
x = tf.placeholder(tf.float32, [None, gorulebilen_katman_boyutu], name="x")
# köşe ağırlıklarını depolayan ağırlık matrisi
W = tf.Variable(tf.random_normal([gorulebilen_katman_boyutu, gizli_katman_boyutu], 0.01), name="W")
# saklı katman için bias vektörü
bh = tf.Variable(tf.zeros([1,gizli_katman_boyutu], tf.float32, name="bh"))
# görülebilen katman için bias vektörü
bv = tf.Variable(tf.zeros([1, gorulebilen_katman_boyutu], tf.float32, name="bv"))


# yardımcı fonksiyonlar.
# bu fonksiyon kolayca olasılıkların vektörünün birini örnekler
def orneklem(probs):
    # Olasılıklardan birinin vektörünü alır,
    # giriş vektöründen örneklenmiş sıfırlar ve birlerin rastgele bir vektörünü döndürür
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# Bu fonksiyon gibbs zincirini koşar. Bu fonk. iki yerde çağıracağız:
#    - eğitimde bir adımı güncellerken
#    - eğitilmiş olan RBM'den müziklerimizi örneklerken
def gibbs_orneklemi(k):
    # W, bh, bv ile tanımlanan RBM'nin olasılık dağılımından bir örnekleme için k-adımlı gibbs zincirini koşar
    def gibbs_adimi(sayac, k, xk):
        # tek bir gibbs adımı koşar. görülebilen değerler xk'ya gider
        hk = orneklem(tf.sigmoid(tf.matmul(xk, W) + bh))  # saklı değerleri örneklemek için görünür değerleri yayar
        xk = orneklem(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # görünür değerleri örneklemek için gizli değerleri yayar
        return sayac + 1, k, xk

    # k iterasyon için gibbs adımları koş
    ct = tf.constant(0)  # counter
    [_, _, x_orneklemi] = control_flow_ops.while_loop(lambda sayac, iter_sayisi, *args: sayac < iter_sayisi,
                                                   gibbs_adimi, [ct, tf.constant(k), x])
    # implemantasyonda illa olmak zorunda değil
    # ama kodu tensorflow'un optimizerlerini kullanarak
    # adapte etme istiyorsak gibbs adımları gerçekleşirken tensor'u durdururuz
    x_orneklemi = tf.stop_gradient(x_orneklemi)
    return x_orneklemi



# Eğitim yükleme kodu
# Şimdi karşıt sapma algoritmasını uyguluyoruz.
# Öncelikle olasılık dağılımındaki x ve h değerlerini alıyoruz
# x örneklemesi
x_orneklemi = gibbs_orneklemi(1)
# x'in görünür değerinden başlayarak saklı düğümlerin örneklemesi
h = orneklem(tf.sigmoid(tf.matmul(x, W) + bh))
# x_orneklemi'ın görünür değerinden başlayarak saklı düğümlerin örneklemesi
h_orneklemi = orneklem(tf.sigmoid(tf.matmul(x_orneklemi, W) + bh))

# çizmiş olduğumuz örnekler arasındaki farka ve orjinal değerlere bağlı olarak
# W, bh, ve bv değerlerini güncelliyoruz
bt_boyutu = tf.cast(tf.shape(x)[0], tf.float32)
W_toplayici = tf.multiply(derece / bt_boyutu,
                      tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_orneklemi), h_orneklemi)))
bv_toplayici = tf.multiply(derece / bt_boyutu, tf.reduce_sum(tf.subtract(x, x_orneklemi), 0, True))
bh_toplayici = tf.multiply(derece / bt_boyutu, tf.reduce_sum(tf.subtract(h, h_orneklemi), 0, True))
# sess.run(updt) yaptığımızda, TensorFlow 3 güncelleme adımını birden koşacak
updt = [W.assign_add(W_toplayici), bv.assign_add(bv_toplayici), bh.assign_add(bh_toplayici)]

# grafiği koş
# bir session başlatıp grafi koşma kısmı burası

with tf.Session() as sess:
    # öncelikle modelimizi eğitiyoruz
    # modelimizin değerlerini başlatıyoruz
    init = tf.global_variables_initializer()
    sess.run(init)
    # epochlarin_sayisi kere tüm eğitim verisi boyunca koş
    for epoch in tqdm(range(epochlarin_sayisi)):
        for sarki in sarkilar:
            # şarkılar zamanda x nota şeklinde depolandı. her şarkının boyutu: "timesteps_in_song x 2*nota_araligi"
            # her eğitim örneği için yeniden şekillendiriyoruz
            #  zaman_adimlari_sayisi x 2*nota_araligi elemanlı bir vektör
            sarki = np.array(sarki)
            sarki = sarki[:int(np.floor(sarki.shape[0] // zaman_adimlari_sayisi) * zaman_adimlari_sayisi)]
            sarki = np.reshape(sarki, [sarki.shape[0] // zaman_adimlari_sayisi, sarki.shape[1] * zaman_adimlari_sayisi])
            # her an ornek_boyutu örneklerinde RBM'yi eğit
            for i in range(1, len(sarki), batch_boyutu):
                tr_x = sarki[i:i + batch_boyutu]
                sess.run(updt, feed_dict={x: tr_x})

    # model şimdi tamamen eğitildi, şimdi biraz müzik yapalım
    # görünür düğümlerin başlangıç değeri 0'a setlendiğinde gibbs zincirini koş
    orneklem = gibbs_orneklemi(1).eval(session=sess, feed_dict={x: np.zeros((10, gorulebilen_katman_boyutu))})
    for i in range(orneklem.shape[0]):
        if not any(orneklem[i, :]):
            continue
        # zamanda x nota olacak şekilde vektörü şekillendirdik ve dosyayı midi olarak kaydettik
        S = np.reshape(orneklem[i, :], (zaman_adimlari_sayisi, 2 * nota_araligi))
        midi_manipulation.noteStateMatrixToMidi(S, "out/olusturulan akor{}".format(i))
