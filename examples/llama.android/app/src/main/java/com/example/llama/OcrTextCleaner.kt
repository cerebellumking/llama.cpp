package com.example.llama

fun cleanOcrText(allText: String): String {
    return allText
        .replace(Regex("""\s*(?:[\u4e00-\u9fa5]{1,6})?(医师|医生|技师)[:：]?\s*[\u4e00-\u9fa5]{2,4}""")) { _ -> "" }
        .replace(Regex("""名[:：]?\s*[\u4e00-\u9fa5]{2,4}"""), "")
        .replace(Regex("""号[:：]?\s*\d+"""), "")
        .replace(Regex("""姓名[:：]?\s*[\u4e00-\u9fa5]{2,4}"""), "")
        .replace(Regex("""患者[:：]?\s*[\u4e00-\u9fa5]{2,4}"""), "")
        .replace(Regex("\n+"), " ")
        .replace(Regex("\\d{17}[\\dXx]"), "")
        .replace(Regex("1[3-9]\\d{9}"), "")
        .replace(Regex("地址[:：]?[\\u4e00-\\u9fa5A-Za-z0-9\\-]{4,}"), "")
        .replace(Regex("[\\u4e00-\\u9fa5]{2,20}医院"), "")
        .replace(Regex("\\s+"), " ")
        .replace(Regex("[^\\p{L}\\p{N}\\p{P}\\s]"), "")
        .replace("姓", "")
        .trim()
}
