CLASS z_humaneval_079 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: decimal_to_binary
      IMPORTING
        VALUE(decimal) TYPE i
      RETURNING
        VALUE(result)  TYPE string.

ENDCLASS.

CLASS z_humaneval_079 IMPLEMENTATION.

  METHOD decimal_to_binary.
    DATA: lv_binary    TYPE string,
          lv_temp      TYPE i,
          lv_remainder TYPE i.

    IF decimal = 0.
      result = 'db0db'.
      RETURN.
    ENDIF.

    lv_temp = decimal.

    WHILE lv_temp > 0.
      lv_remainder = lv_temp MOD 2.
      lv_temp = lv_temp DIV 2.
      lv_binary = |{ lv_remainder }{ lv_binary }|.
    ENDWHILE.

    result = |db{ lv_binary }db|.
  ENDMETHOD.

ENDCLASS.