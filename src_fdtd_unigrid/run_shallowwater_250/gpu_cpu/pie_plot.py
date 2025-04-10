import matplotlib.pyplot as plt


plt.figure(figsize=(12,8))

plt.subplot(121)

labels = ['compute curlE', 'update H', 'compute curlH', 'inject J', 'update E', 'DFT','Convergence']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red','yellow','blue']

time = [2.469823e+02, 8.615623e+01, 2.247243e+02, 6.892280e-03,8.510836e+01,7.986012e+01,2.176370e-01]
patches, texts = plt.pie(time, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.title('(a) CPU: omp=1')


plt.subplot(122)

time = [1.814154e+00, 3.463412e+00, 2.416798e+00, 1.375645e-02,6.491812e+00,9.833440e-03,3.839466e+00]
patches, texts = plt.pie(time, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.title('(b) GPU')

plt.savefig('time_distribution.png')
plt.show()
