pipeline {
  agent any

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr: '10'))
  }

  stages {
    stage('Validate') {
      steps {
        script {
          def homepage = readFile('src/index.html')
          if (!homepage.contains('Jenkins B2')) {
            error('Expected project title not found in src/index.html')
          }

          def stylesheet = readFile('src/styles.css')
          if (!stylesheet.contains(':root')) {
            error('Expected CSS variables not found in src/styles.css')
          }
        }
      }
    }

    stage('Test') {
      steps {
        script {
          def app = readFile('src/app.js')
          if (!app.contains('new Date')) {
            error('Expected build time rendering logic not found in src/app.js')
          }
        }
      }
    }

    stage('Package') {
      steps {
        script {
          if (isUnix()) {
            sh 'rm -rf dist && mkdir -p dist && cp -R src/. dist/'
          } else {
            bat 'if exist dist rmdir /S /Q dist'
            bat 'mkdir dist'
            bat 'xcopy src dist /E /I /Y'
          }
        }
      }
    }
  }

  post {
    success {
      archiveArtifacts artifacts: 'dist/**', fingerprint: true
    }
    always {
      echo "Pipeline finished with status: ${currentBuild.currentResult}"
    }
  }
}
