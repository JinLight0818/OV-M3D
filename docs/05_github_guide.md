# GitHub дээр төслөө байршуулах

## Орон нутгийн түүх

Git (хувилбарын түүх хөтлөх хэрэгсэл) нь төслийн үндэс дэх `.git` хавтсанд түүхээ хадгална. Терминалд төслийн хавтсаа нээгээд дараахыг ажиллуулна. Бүртгэлтэй хуулбарын төлөвийг эхэлж үзнэ.

```bash
pwd
git status
```

Хэрэв эх кодыг архиваас задалсан бол эхний салааг үүсгэнэ.

```bash
git init -b main
```

Зохиогчийн нэр ба цахим шуудан тохируулаагүй бол энэ репозиторт хэрэглэх утгаа оруулна. GitHub-ийн нууцалсан шуудангийн хаягийг мөн ашиглаж болно.

```bash
read -rp 'Бүртгэлд харагдах нэрээ оруулна уу. ' author_name
read -rp 'Бүртгэлд харагдах шуудангаа оруулна уу. ' author_email
git config user.name "$author_name"
git config user.email "$author_email"
git status --short
git add .
git diff --cached --stat
git diff --cached --check
```

Conventional Commits (өөрчлөлтийн төрлийг эхэнд нь заах бичлэгийн дүрэм)-д `feat` нь шинэ боломж, `fix` нь алдаа засвар, `docs` нь баримт, `test` нь шалгалт. Дараах тушаалыг бүртгэх өөрчлөлттэй үед ажиллуулна. [Дүрмийн эх](https://www.conventionalcommits.org/en/v1.0.0/)

```bash
git commit -m "feat: камер ба зайн мэдрэгчийн хэмжилтийг нэгтгэв"
```

## GitHub дээр шинээр үүсгэх

GitHub CLI (GitHub-ийн командын хэрэгсэл)-аар бүртгэлдээ нэвтэрч эзэмшигчээ шалгана. Дараах үүсгэх үйлдэл эх кодыг нийтэд нээлттэй болгоно. Хувийн репозитор хүсвэл `--private` гэж сонгоно. [Репозитор үүсгэх албан заавар](https://cli.github.com/manual/gh_repo_create)

```bash
sudo apt install gh
gh auth login
gh auth status
gh repo create OV-M3D --public --source=. --remote=origin --push
```

Хэрэв GitHub-ийн вэб хуудсаар хоосон репозитор үүсгэсэн бол өмнөх `gh repo create` тушаалын оронд SSH (түлхүүрээр баталгаажуулсан холболт)-ийн замыг хэрэглэнэ. Хаягийг репозиторын `Code → SSH` цэснээс хуулна. Компьютерын нийтийн түлхүүрийг GitHub бүртгэлдээ урьдчилан холбосон байна. [SSH холбох заавар](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

```bash
read -rp 'Репозиторын SSH хаягийг оруулна уу. ' repo_ssh
git remote add origin "$repo_ssh"
git remote -v
git push -u origin main
```

Дараагийн өөрчлөлт бүрд шалгалтаа ажиллуулж, холбогдох файлуудаа бүртгэн `git push` хийнэ. Загварын том жин, камерын хувийн бичлэг, төхөөрөмж дээр хийсэн хэмжилтүүдийг `.gitignore` бүртгэлээс хасдаг.

## Хувилбар тэмдэглэж нийтлэх

Доорх жишээ эхний хувилбарыг тэмдэглэнэ. Tag (түүхийн тодорхой цэгийн нэр) болон Release (тэмдэглэгээнд холбосон нийтлэл) нь ижил шалгагдсан бүртгэл рүү заана. [Хувилбар нийтлэх заавар](https://cli.github.com/manual/gh_release_create)

```bash
python -m pytest -q
git status --short
git tag -a v0.1.0 -m "Хэмжилтийн анхны хувилбар"
git push origin v0.1.0
gh release create v0.1.0 --verify-tag --title "OV-M3D v0.1.0" --notes "Камер ба мэдрэгчийн хэмжилт, калибровк, үнэлгээний хэрэгслүүд."
```
