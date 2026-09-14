-keep class com.arm.aichat.* { *; }
-keep class com.arm.aichat.gguf.* { *; }

-keepclasseswithmembernames class * {
    native <methods>;
}

-keep class kotlin.Metadata { *; }

# JNI in llama_compat.cpp calls these by name on Llm-RunLoop.
-keep class android.llama.cpp.LLamaAndroid { *; }
-keep class android.llama.cpp.LLamaAndroid$Companion { *; }
-keep class android.llama.cpp.LLamaAndroid$Companion$IntVar {
    <init>(int);
    public void inc();
    public int getValue();
}
-keepclassmembers class android.llama.cpp.LLamaAndroid$Companion$IntVar {
    public void inc();
    public int getValue();
}
