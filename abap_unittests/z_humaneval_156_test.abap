CLASS ltcl_test_int_to_mini_roman DEFINITION
FOR TESTING
DURATION SHORT
RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_19 FOR TESTING,
      test_152 FOR TESTING,
      test_251 FOR TESTING,
      test_426 FOR TESTING,
      test_500 FOR TESTING,
      test_1 FOR TESTING,
      test_4 FOR TESTING,
      test_43 FOR TESTING,
      test_90 FOR TESTING,
      test_94 FOR TESTING,
      test_532 FOR TESTING,
      test_900 FOR TESTING,
      test_994 FOR TESTING,
      test_1000 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_int_to_mini_roman IMPLEMENTATION.
  METHOD test_19.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 19 )
      exp = 'xix' ).
  ENDMETHOD.

  METHOD test_152.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 152 )
      exp = 'clii' ).
  ENDMETHOD.

  METHOD test_251.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 251 )
      exp = 'ccli' ).
  ENDMETHOD.

  METHOD test_426.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 426 )
      exp = 'cdxxvi' ).
  ENDMETHOD.

  METHOD test_500.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 500 )
      exp = 'd' ).
  ENDMETHOD.

  METHOD test_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 1 )
      exp = 'i' ).
  ENDMETHOD.

  METHOD test_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 4 )
      exp = 'iv' ).
  ENDMETHOD.

  METHOD test_43.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 43 )
      exp = 'xliii' ).
  ENDMETHOD.

  METHOD test_90.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 90 )
      exp = 'xc' ).
  ENDMETHOD.

  METHOD test_94.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 94 )
      exp = 'xciv' ).
  ENDMETHOD.

  METHOD test_532.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 532 )
      exp = 'dxxxii' ).
  ENDMETHOD.

  METHOD test_900.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 900 )
      exp = 'cm' ).
  ENDMETHOD.

  METHOD test_994.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 994 )
      exp = 'cmxciv' ).
  ENDMETHOD.

  METHOD test_1000.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_156=>int_to_mini_roman( 1000 )
      exp = 'm' ).
  ENDMETHOD.
ENDCLASS.