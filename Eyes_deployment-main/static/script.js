// Image Preview
const imageUpload = document.getElementById("imageUpload");
const previewImage = document.getElementById("previewImage");
const uploadBox = document.getElementById("uploadBox");
const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
const submitBtn = document.getElementById("submitBtn");

// When user clicks the upload area
uploadBox.addEventListener("click", () => {
  imageUpload.click();
});

// Show preview after selecting file
imageUpload.addEventListener("change", function () {
  const file = this.files[0];

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }
});

// Prevent submitting without image
form.addEventListener("submit", function (e) {
  if (!imageUpload.value) {
    alert("Kindly upload an image before prediction dear user.");
    e.preventDefault();
    return;
  }

  loader.classList.remove("hidden"); // show loader
  submitBtn.disabled = true;
});

// Dark Mode Toggle
const toggleSwitch = document.getElementById("darkModeSwitch");
toggleSwitch.addEventListener("change", () => {
  document.body.classList.toggle("dark-mode");
  localStorage.setItem("darkMode", toggleSwitch.checked);
});

// Keep mode saved on page reload
if (localStorage.getItem("darkMode") === "true") {
  document.body.classList.add("dark-mode");
  toggleSwitch.checked = true;
}
