CLASS z_humaneval_076 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_simple_power
      IMPORTING
        VALUE(x)      TYPE i
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_076 IMPLEMENTATION.

  METHOD is_simple_power.
    DATA: power TYPE i.

    IF n = 1.
      IF x = 1.
        result = abap_true.
      ELSE.
        result = abap_false.
      ENDIF.
      RETURN.
    ENDIF.

    power = 1.
    WHILE power < x.
      power = power * n.
    ENDWHILE.

    IF power = x.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.