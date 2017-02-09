# detect author gender from essay corpus

#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import feature_extractor as fe
import re
import sys
import time
start_time = time.time()
'''
datadir = sys.argv[1]
genderlabfile = datadir + "/BAWE_balanced_subset.csv"
conffile = sys.argv[2]
outfile = sys.argv[3]
'''
datadir = "CORPUS_TXT"
genderlabfile = datadir + "/BAWE_balanced_subset.csv"
conffile = "conf.txt"
outfile = "features.csv"

def load_balanced_gender_labels():
	'''
	Read the gender labels file and return dictionary mapping student id
	to gender
	'''
	meta_lines = [line.rstrip().split(',') for line in open(genderlabfile)]
	gender_dict = {row[0]:row[1] for row in meta_lines[1:]}
	return gender_dict

def load_essays(gender_dict):
	essays = []
	genderlabels = []
	students = []
	for student, gender in gender_dict.items():
		with open('%s/%s.txt' % (datadir, student)) as f:
			text = f.read().decode('utf-8')
			text = re.sub('<[^<]+?>', '', text)		# remove xml
			essays.append(text)
			genderlabels.append(gender)
			students.append(student)
	return essays, genderlabels, students

def load_conf_file():
	conf = set(line.strip() for line in open(conffile))
	return conf

def predict_gender(X, Y):
	scores = cross_val_score(GaussianNB(), X, Y, scoring='accuracy', cv=10)
	return scores.mean()

def write_features(features,header,genderlabels,students):
	str_gender_labels = ['male' if g is '0' else 'female' for g in genderlabels]
	with open(outfile,'w') as f:
		f.write('student_id,' + ','.join(header)+ ',gender\n')
		for student,line,gender in zip(students,features,str_gender_labels):
			str_line = [str(val) for val in line]

			f.write(student+ ',' + ','.join(str_line)+','+gender+'\n')
		f.close()

if __name__ == "__main__":
	gender_dict = load_balanced_gender_labels()
	essays, genderlabels, students = load_essays(gender_dict)
	conf = load_conf_file()
	features,header = fe.extract_features(essays, conf)
	write_features(features,header,genderlabels,students)
	print (predict_gender(features, genderlabels))
	print("--- %s seconds ---" % (time.time() - start_time))