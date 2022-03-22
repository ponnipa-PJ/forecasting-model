<?php
$target_dir = "data/";
$target_file = $target_dir . basename($_FILES["afile"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
if(isset($_POST["submit"])) {
    if (move_uploaded_file($_FILES["afile"]["tmp_name"], $target_file)) {
        // echo "ไฟล์ ". htmlspecialchars( basename( $_FILES["afile"]["name"])). "ถูก upload เรียบร้อยแล้ว";
        $myfile = fopen("filename.txt", "w") or die("Unable to open file!");
        $txt =  $target_dir . htmlspecialchars( basename( $_FILES["afile"]["name"]));
        fwrite($myfile, $txt);
        fclose($myfile);
        header( "location: http://127.0.0.1:5000/" );
        exit(0);
      } else {
        echo " ไม่สามารถอ่านไฟล์ได้";
      }
}
?>