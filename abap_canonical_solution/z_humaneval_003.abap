CLASS z_humaneval_003 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS below_zero
      IMPORTING
        VALUE(operations) TYPE int4_table
      RETURNING
        VALUE(result)     TYPE abap_bool.
  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_003 IMPLEMENTATION.

  METHOD below_zero.
    DATA(balance) = 0.

    LOOP AT operations INTO DATA(op).
      balance += op.
      IF balance < 0.
        result = abap_true.
        RETURN.
      ENDIF.
    ENDLOOP.

    result = abap_false.
  ENDMETHOD.

ENDCLASS.