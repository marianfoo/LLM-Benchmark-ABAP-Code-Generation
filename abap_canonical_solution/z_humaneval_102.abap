CLASS z_humaneval_102 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: choose_num
      IMPORTING
        !x            TYPE i
        !y            TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_102 IMPLEMENTATION.

  METHOD choose_num.
    IF x > y.
      result = -1.
      RETURN.
    ENDIF.

    IF y MOD 2 = 0.
      result = y.
      RETURN.
    ENDIF.

    IF x = y.
      result = -1.
      RETURN.
    ENDIF.

    result = y - 1.
  ENDMETHOD.

ENDCLASS.