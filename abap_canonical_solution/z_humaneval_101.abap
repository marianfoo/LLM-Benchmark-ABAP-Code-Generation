CLASS z_humaneval_101 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: words_string
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE string_table.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_101 IMPLEMENTATION.
  METHOD words_string.
    DATA: lv_temp TYPE string.

    IF s IS INITIAL.
      RETURN.
    ENDIF.

    lv_temp = s.
    REPLACE ALL OCCURRENCES OF ',' IN lv_temp WITH ' '.

    SPLIT lv_temp AT ' ' INTO TABLE result.
    DELETE result WHERE table_line IS INITIAL.
  ENDMETHOD.
ENDCLASS.