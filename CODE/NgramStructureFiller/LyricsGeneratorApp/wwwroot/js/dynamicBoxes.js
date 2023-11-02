function GetDynamicTextBox(value) {
    var div = $("<div />");

    var textBox = $("<input />").attr("type", "textbox").attr("name", "DynamicTextBox");
    textBox.val(value);
    div.append(textBox);

    var button = $("<input />").attr("type", "button").attr("value", "Remove");
    button.attr("onclick", "RemoveTextBox(this)");
    div.append(button);

    return div;
}
function AddTextBox() {
    var div = GetDynamicTextBox("");
    $("#TextBoxContainer").append(div);
}

function RemoveTextBox(button) {
    $(button).parent().remove();
}

$(function () {
    var values = eval('@Html.Raw(ViewBag.Values)');
    if (values != null) {
        $("#TextBoxContainer").html("");
        $(values).each(function () {
            $("#TextBoxContainer").append(GetDynamicTextBox(this));
        });
    }
});