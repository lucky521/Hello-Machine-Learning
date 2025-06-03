---
layout: post
title:  "Apple Machine Learning Journal"
subtitle: ""
categories: [MachineLearning]
---

这篇博客解读了Apple公司的机器学习博客里的文章。https://machinelearning.apple.com/
本Blog中的论文主要跟语音、语义、图像识别有关。

# Improving the Realism of Synthetic Images

图像识别领域里，模型训练的一个困难点在于缺少足够多的具备label的真实图像数据集。生成合成图像样本是一个方法，但是对合成图像的质量有较高要求，必须要足够接近真实样本的分布，否则将会误导模型训练的走向。这篇文章设计了一种方法来提高合成图像的质量，使得模型在真实应用时具有满意的泛化能力。

# Improving Neural Network Acoustic Models by Cross-bandwidth and Cross-lingual Initialization

语音识别领域里，同样缺乏特定语言下的标签样本数据。这篇文章设计了迁移学习方法，从训练好的另一种语言的Acoustic Model迁移数据。

# Inverse Text Normalization as a Labeling Problem

语音转文字领域里，inverse text normalization (ITN)是要把语音转文本时遇到的日期、时间、地点、价钱等内容以合理的形式显示出来。这篇文章把这一个问题当做Labeling Problem，用统计学模型来解决。

# Deep Learning for Siri’s Voice: On-device Deep Mixture Density Networks for Hybrid Unit Selection Synthesis

人声语音合成领域里，有两种技术，unit selection synthesis 和 parametric synthesis。unit selection synthesis在具备足够多的高质量素材的情况下能够提供高质量的输出。parametric synthesis在具备少量素材的情况下能够提供流畅易懂的输出。hybrid system指的就是两者的结合，使用parametric approach来进行unit selection，称作Hybrid unit selection methods。这篇文章介绍了使用深度学习技术为Siri实现更自然的语音合成。


# Real-Time Recognition of Handwritten Chinese Characters Spanning a Large Inventory of 30,000 Characters

在手写输入识别中，汉字等符号类文字的识别可以借助深度学习达到极高的准确率。

# Hey Siri: An On-device DNN-powered Voice Trigger for Apple’s Personal Assistant

语音识别领域里，Siri支持的hey siri功能需要一个小的speech detector来持续接收声音并处理。它的要求是要以最低的功耗监听和识别出hey siri这个词。

# An On-device Deep Neural Network for Face Detection

人脸识别技术，已经被应用到手机设备上，无需依赖网络服务器。在Apple的图像架构基础API中CIDetector提供了人脸识别功能，可以用所有APP调用。早期的版本使用的是Viola-Jones算法。这篇文章介绍了如何利用深度学习方法实现更好的效果。

# Learning with Privacy at Scale

移动设备为了提供更好的使用体验，需要收集用户的使用习惯数据。为了在数据收集和隐私保护之间平衡折中，这篇文章设计了一套学习系统。

# Personalized Hey Siri

人声识别时语音识别中的一个分支。speaker recognition的核心目标不是判断语音的内容是什么，而是要判断是不是目标人物的声音。

# Finding Local Destinations with Siri’s Regionally Specific Language Models for Speech Recognition

语音识别领域里，Siri借助地理位置信息来增强语音识别的效果，因为许多语音素材，比如地名，和地域范围有一定联系。

# Can Global Semantic Context Improve Neural Language Models?

输入预测是NLP的一个应用场景，苹果设备中的QuickType keyboard支持在多种App下进行输入预测。本篇文章介绍了其所使用global semantic context来训练word embedding和NLP model。
