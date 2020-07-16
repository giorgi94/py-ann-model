fetch("/api/test/")
    .then((r) => r.json())
    .then((data) => {
        console.log(data);
    });
