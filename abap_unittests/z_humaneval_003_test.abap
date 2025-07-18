CLASS z_humaneval_003_demo DEFINITION
  FOR TESTING
  FINAL
  CREATE PUBLIC
  RISK LEVEL HARMLESS.

  PUBLIC SECTION.
  PRIVATE SECTION.
    METHODS:
      test_empty_table FOR TESTING,
      test_balanced_operations FOR TESTING,
      test_goes_below_zero FOR TESTING,
      test_alternating_balanced FOR TESTING,
      test_alternating_goes_below FOR TESTING,
      test_early_negative_balance FOR TESTING.
ENDCLASS.

CLASS z_humaneval_003_demo IMPLEMENTATION.

  METHOD test_empty_table.
    DATA: lv_empty_table TYPE int4_table.

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lv_empty_table )
      exp = abap_false
      msg = 'Empty table should return false - no operations means balance never goes below zero' ).
  ENDMETHOD.

  METHOD test_balanced_operations.
    DATA: lt_operations TYPE int4_table.

    lt_operations = VALUE int4_table( ( 1 ) ( 2 ) ( -3 ) ( 1 ) ( 2 ) ( -3 ) ).

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lt_operations )
      exp = abap_false
      msg = 'Balanced operations should return false - balance never goes below zero' ).
  ENDMETHOD.

  METHOD test_goes_below_zero.
    DATA: lt_operations TYPE int4_table.

    lt_operations = VALUE int4_table( ( 1 ) ( 2 ) ( -4 ) ( 5 ) ( 6 ) ).

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lt_operations )
      exp = abap_true
      msg = 'Should return true - balance goes below zero after third operation (1+2-4=-1)' ).
  ENDMETHOD.

  METHOD test_alternating_balanced.
    DATA: lt_operations TYPE int4_table.

    lt_operations = VALUE int4_table( ( 1 ) ( -1 ) ( 2 ) ( -2 ) ( 5 ) ( -5 ) ( 4 ) ( -4 ) ).

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lt_operations )
      exp = abap_false
      msg = 'Alternating balanced operations should return false - balance stays at or above zero' ).
  ENDMETHOD.

  METHOD test_alternating_goes_below.
    DATA: lt_operations TYPE int4_table.

    lt_operations = VALUE int4_table( ( 1 ) ( -1 ) ( 2 ) ( -2 ) ( 5 ) ( -5 ) ( 4 ) ( -5 ) ).

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lt_operations )
      exp = abap_true
      msg = 'Should return true - final operation makes balance go below zero' ).
  ENDMETHOD.

  METHOD test_early_negative_balance.
    DATA: lt_operations TYPE int4_table.

    lt_operations = VALUE int4_table( ( 1 ) ( -2 ) ( 2 ) ( -2 ) ( 5 ) ( -5 ) ( 4 ) ( -4 ) ).

    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_003=>below_zero( lt_operations )
      exp = abap_true
      msg = 'Should return true - balance goes below zero early (1-2=-1)' ).
  ENDMETHOD.

ENDCLASS.