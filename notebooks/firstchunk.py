import re
def fixe_size_chunk(para):
    result=[]
    t=0
    temp=''
    for i in para:
        if t==100:
            t=0
            result.append(temp)
            temp=''
        else:
            temp+=i
            t+=1
    return result
            
def word_splitter(para):
    result=para.split()
    return result

def sentence_splitter(para):
    result= re.split(r'.!?',para)
    return result
def newline_splitter(para):
    return para.split(seprator='\n')
def para_splitter(para):
    return para.split(seprator='\n\n')
    
para="""Artificial intelligence (AI) is technology that enables computers and machines to simulate human learning, comprehension, problem solving, decision making, creativity and autonomy.

Applications and devices equipped with AI can see and identify objects. They can understand and respond to human language. They can learn from new information and experience. They can make detailed recommendations to users and experts. They can act independently, replacing the need for human intelligence or intervention (a classic example being a self-driving car).

But in 2024, most AI researchers, practitioners and most AI-related headlines are focused on breakthroughs in generative AI (gen AI), a technology that can create original text, images, video and other content. To fully understand generative AI, it’s important to first understand the technologies on which generative AI tools are built: machine learning (ML) and deep learning.

Machine learning
A simple way to think about AI is as a series of nested or derivative concepts that have emerged over more than 70 years:Directly underneath AI, we have machine learning, which involves creating models by training an algorithm to make predictions or decisions based on data. It encompasses a broad range of techniques that enable computers to learn from and make inferences based on data without being explicitly programmed for specific tasks.

There are many types of machine learning techniques or algorithms, including linear regression, logistic regression, decision trees, random forest, support vector machines (SVMs), k-nearest neighbor (KNN), clustering and more. Each of these approaches is suited to different kinds of problems and data.

But one of the most popular types of machine learning algorithm is called a neural network (or artificial neural network). Neural networks are modeled after the human brain's structure and function. A neural network consists of interconnected layers of nodes (analogous to neurons) that work together to process and analyze complex data. Neural networks are well suited to tasks that involve identifying complex patterns and relationships in large amounts of data.

The simplest form of machine learning is called supervised learning, which involves the use of labeled data sets to train algorithms to classify data or predict outcomes accurately. In supervised learning, humans pair each training example with an output label. The goal is for the model to learn the mapping between inputs and outputs in the training data, so it can predict the labels of new, unseen data."""


print(len(fixe_size_chunk(para)))
print(len(word_splitter(para)))
print(len(sentence_splitter(para)))
print(len(para_splitter(para)))
print(para_splitter(para))
