CLASS z_humaneval_067 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: fruit_distribution
      IMPORTING
        s             TYPE string
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_067 IMPLEMENTATION.

  METHOD fruit_distribution.
    DATA: numbers    TYPE TABLE OF i,
          word_table TYPE TABLE OF string,
          word       TYPE string,
          sum        TYPE i.

    SPLIT s AT ' ' INTO TABLE word_table.

    LOOP AT word_table INTO word.
      IF word CO '0123456789'.
        APPEND CONV i( word ) TO numbers.
      ENDIF.
    ENDLOOP.

    LOOP AT numbers INTO DATA(number).
      sum = sum + number.
    ENDLOOP.

    result = n - sum.

  ENDMETHOD.

ENDCLASS.