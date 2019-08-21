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


