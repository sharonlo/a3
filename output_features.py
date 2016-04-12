import csv
import numpy
import scipy
from scipy import stats
from collections import OrderedDict


def features(raw, feature_path):
	writeHeader = False
	r = open (raw, 'r')
	reader = csv.DictReader(r)

	with open (feature_path, 'w') as w:
		fast_fourier = []
		mean = {}
		stDev = {}
		fft = {}
		fft_stDev = {}

		#Iterate through csv and append values to appropriate feature arrays
		for row_num, row in enumerate(reader):
			if row_num == 0:
				start = int(row['timestamp'])
				end = start + 10
			if float(row['timestamp']) < end:
				for k, v in row.iteritems():
					if k != 'timestamp':
						if "'"+k+"_mean'" not in mean:
							mean["'"+k+"_mean'"] = [float(v)]
							stDev["'"+k+"_stDev'"] = [float(v)]
							fft["'"+k+"_fft'"] = [float(v)]
							fft_stDev["'"+k+"_fft_stDev'"] = [float(v)]
						else:
							mean["'"+k+"_mean'"].append(float(v))
							stDev["'"+k+"_stDev'"].append(float(v))
							fft["'"+k+"_fft'"].append(float(v))
							fft_stDev["'"+k+"_fft_stDev'"].append(float(v))

			else:
				for k, v in mean.iteritems():
					mean[k] = numpy.mean(numpy.array(mean[k]))
				for k, v in stDev.iteritems():
					stDev[k] = numpy.std(numpy.array(stDev[k]))
				for k, v in fft.iteritems():
				 	fft[k] = max(abs(scipy.fft(fft[k])).tolist())
				for k, v in fft_stDev.iteritems():
				 	fft_stDev[k] = numpy.std(numpy.array((scipy.fft(fft_stDev[k]).tolist())))
					
				a = merge(mean, stDev)
				b = merge(fft, fft_stDev)
				c = merge(a, b)
				mergeDict = OrderedDict(sorted(c.items(), key=lambda t: t[0]))
				writer= csv.DictWriter(w, mergeDict.keys())
				if writeHeader == False:
					writer.writeheader()
					writeHeader = True
				writer.writerow(mergeDict)
				end += 10
				mean = {}
				stDev = {}
				fft = {}
				fft_stDev = {}

def merge(x, y):
    z = x.copy()
    z.update(y)
    return z

features('data/original/final_running.csv', 'data/running.csv')
features('data/original/final_walking.csv', 'data/walking.csv')
features('data/original/final_sitting.csv', 'data/sitting.csv')
features('data/original/final_stairs.csv', 'data/stairs.csv')
features('data/original/final_biking.csv', 'data/biking.csv')

