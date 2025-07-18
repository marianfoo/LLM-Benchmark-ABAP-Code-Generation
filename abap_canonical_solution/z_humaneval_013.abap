CLASS z_humaneval_013 DEFINITION
    PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS:
      greatest_common_divisor
        IMPORTING
          VALUE(a)      TYPE i
          VALUE(b)      TYPE i
        RETURNING
          VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_013 IMPLEMENTATION.

  METHOD greatest_common_divisor.
    WHILE b <> 0.
      DATA(lv_temp) = a.
      a = b.
      b = lv_temp MOD b.
    ENDWHILE.
    result = a.
  ENDMETHOD.

ENDCLASS.