const buildTime = new Date();

document.getElementById("buildTime").innerText =
    "Build Time: " + buildTime.toLocaleString();

console.log("Application Loaded");