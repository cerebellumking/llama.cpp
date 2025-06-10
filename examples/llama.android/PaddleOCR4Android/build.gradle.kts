plugins {
    id("com.android.library")
    id("kotlin-android")
    id("maven-publish")
}

android {
    namespace = "android.PaddleOCR4Android"
    compileSdk = 34
    ndkVersion = "21.1.6352462"
    defaultConfig {
       minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++11 -frtti -fexceptions -Wno-format")
                arguments(
                    "-DANDROID_PLATFORM=android-23",
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_ARM_NEON=TRUE"
                )
            }
        }
        ndk {
            abiFilters += "arm64-v8a"
            abiFilters += "armeabi-v7a"
            ldLibs?.plusAssign("jnigraphics")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// afterEvaluate {
//     publishing {
//         publications {
//             create<MavenPublication>("release") {
//                 from(components["release"])
//                 groupId = "com.equationl.paddleocr4android"
//                 artifactId = "paddleocr4android"
//                 version = "V1.2.0"
//             }
//         }
//     }
// }

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.5.0")
    testImplementation("junit:junit:4.+")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
