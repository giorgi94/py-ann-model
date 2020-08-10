const box = document.querySelector(".box");
const btn = document.querySelector("#btn");
const comment = document.querySelector("#comment");

btn.addEventListener("click", () => {
    let data = new FormData();

    data.append("comment", comment.value);

    fetch("/api/test/", {
        method: "POST",
        body: data,
    })
        .then((r) => r.json())
        .then((data) => {
            const label = data.label;
            if (label == "-") {
                box.className = "box red";
            } else if (label == "+") {
                box.className = "box green";
            }
        });
});
