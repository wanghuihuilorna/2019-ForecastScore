# ForecastScore
2019 “添翼”人工智能创新应用大赛-智慧教育通道

一、赛题描述
请参赛选手，利用比赛对应训练集提供的学生信息、考试知识点信息、考试总得分信息等建立模型，预测测试集中学生在指定考试中的成绩总分，预测目标如下：

初赛：利用初中最后一年的相关考试和考点信息，预测初中最后一学期倒数第二、第三次考试的成绩。
复赛：利用初中 4 年中的相关考试和考点信息，预测初中最后一学期最后一次考试的的成绩。


二、训练集
student.csv 学生信息

序列	    包含字段	        格式	        解释说明	            示例
字段1	student_id	    integer 	脱敏后的学生独立标识	389253
字段2	gender	        integer	    学生性别，0和1	    0



course.csv 课程信息

序列	    包含字段	        格式	        解释说明                       示例
字段1	course	        string	    脱敏后课程名称，如语文、数学	     course1
字段2	course_class	string	    脱敏后课程分类，如文科、理科	     course_class1



all_knowledge.csv 知识点信息

序列	    包含字段	            格式	        解释说明	                                                                        示例
字段1	course	            string	    脱敏后课程名称，如物理	                                                            course1
字段2	knowledge_point	    string	    脱敏后课程中所学知识点，如牛顿第三定律在对应的course内唯一，不同course的知识点没有对应关系	    K:0
字段3	section	            string	    脱敏后知识点所属段落，如动力学在对应的course内唯一，不同course的段落没有对应关系	            S:0
字段4	category	        string	    脱敏后知识点所属类目，如力学在对应的course内唯一，不同course的类目没有对应关系	            C:0
字段5	complexity	        integer	    难度等级，如极其容易、 容易、中等、难、极其难	3



course*_exams.csv 试卷信息

序列	    包含字段	    格式	        解释说明	                                                示例
字段1	exam_id	    string  	脱敏后考试独立标识	                                        FJxoUDCI
字段2至n	K:n	        integer 	脱敏后某次考试中每个知识点（K:n)，占的分数，每次考试满分100分	    0；0；0；0；0；0；6；3；0；6；...



exam_score_s1.csv（初赛） 和 exam_score.csv（复赛） 得分信息

序列	    包含字段	        格式	        解释说明	                示例
字段1	student_id	    integer	    脱敏后的学生独立标识	    230748
字段2	course	        string	    脱敏后课程名称，如物理	    course1
字段3	exam_id	        string	    脱敏后考试独立标识	        FJxoUDCI
字段4	score	          integer	    考试总分	                73



三、测试集
submission_s1.csv（初赛） 和 submission_s2.csv （复赛）

序列	    包含字段	        格式	        解释说明	                示例
字段1	student_id	    integer	    脱敏后的学生独立标识	    230748
字段2	course	        string	    脱敏后课程名称，如物理	    course1
字段3	exam_id	        string	    脱敏后考试独立标识	        m31I6cTD
字段4	pred	          float	    考试总分预测



#########################################           改进之处           #########################################

一:
# log1p就是log(1+x)，用来对得分进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
# 需要注意，最后需要将预测出的平滑数据还原，而还原过程就是log1p的逆运算expm1。
train["score"] = np.log1p(train["score"])

二:
# 对于列名为'年'、'月'、'日'的特征列，将列中的数据类型转化为string格式。   [待查找]

三:
######################数字型数据列偏度校正-【开始】#######################
# 使用skew()方法，计算所有整型和浮点型数据列中，数据分布的偏度（skewness）。
# 偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。亦称偏态、偏态系数。
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

# 以0.5作为基准，统计偏度超过此数值的高偏度分布数据列，获取这些数据列的index。
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

# 对高偏度数据进行处理，将其转化为正态分布。
# Box和Cox提出的变换可以使线性回归模型满足线性性、独立性、方差齐次以及正态性的同时，又不丢失信息。
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))  # 这是boxcox1p的使用方法，参数的具体意义暂时不解释
######################数字型数据列偏度校正-【结束】#######################